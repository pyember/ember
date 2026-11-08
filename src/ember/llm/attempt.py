"""Running one model call: attempts, retries, pricing and reports.

A `Call` is built by the client once every check that can fail before sending
has passed and the call holds a provider lease and a rate-limit permit. Its
`run` (or `events`, for streams) is the task handed to the scheduler. It is
the only code in ember that retries:

- only a `ModelError` whose ``retryable`` is true, and at most
  `RetryPolicy.max_attempts` attempts in all;
- after a jittered exponential delay, clamped to `RetryPolicy.max_delay_s`
  after the jitter, and never sooner than the provider's ``retry_after_s``;
- never when the provider asks for a longer wait than ``max_delay_s``;
- never once a stream has delivered its first item: providers mark such
  errors ``before_first_byte=False``, and the client refuses the retry itself
  whatever the error says, since a retry would deliver that output twice.

Every attempt, retries included, is reported to the observers that were
active where the call was made, and to the provider's limiter.
"""

import dataclasses
import time
import weakref
from collections.abc import Callable, Generator, Iterator

from ember.context import RetryPolicy
from ember.errors import ErrorKind, ModelError
from ember.llm.limiter import Permit
from ember.llm.observe import Attempt, Reporter
from ember.llm.pool import Lease
from ember.llm.provider import Decoded
from ember.llm.request import Request
from ember.llm.result import Done, Event, Money, Result
from ember.llm.spec import ModelSpec

__all__ = [
    "PRICE_NEUTRAL_EXTENSIONS",
    "Call",
    "cost_of",
    "extensions_keep_list_price",
    "retry_delay",
]

# Bounds 2**n so a very large attempt count cannot overflow a float.
_MAX_DOUBLINGS = 62


def retry_delay(
    policy: RetryPolicy, attempt: int, error: ModelError, jitter: float
) -> float | None:
    """Returns the seconds to wait before retrying, or ``None`` to stop.

    Args:
        policy: The caller's retry policy.
        attempt: The number of the attempt that just failed, from 1.
        error: Why it failed.
        jitter: A uniform random number in [0, 1).

    Returns:
        ``base_delay_s * 2**(attempt - 1)`` scaled by a factor in [0.5, 1.5),
        clamped to ``max_delay_s``, and raised to ``retry_after_s`` when the
        provider asked for longer. ``None`` when the error is not retryable,
        the attempts are used up, or ``retry_after_s`` exceeds ``max_delay_s``.
    """
    if not error.retryable or attempt >= policy.max_attempts:
        return None
    backoff = policy.base_delay_s * 2.0 ** min(attempt - 1, _MAX_DOUBLINGS) * (0.5 + jitter)
    delay = min(policy.max_delay_s, backoff)
    if error.retry_after_s is None:
        return delay
    if error.retry_after_s > policy.max_delay_s:
        return None
    return max(delay, error.retry_after_s)


#: Extension keys that never change what a provider charges per token: request
#: labels, routing and storage hints, and extra output fields. Any other key may
#: select a different price (a service tier, priority or fast mode, a cache
#: retention period, data residency), so a request carrying one is unpriced.
PRICE_NEUTRAL_EXTENSIONS: frozenset[str] = frozenset(
    {
        "metadata",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "parallel_tool_calls",
        "store",
        "include",
    }
)


def extensions_keep_list_price(request: Request) -> bool:
    """Whether no extension on ``request`` may change what the provider charges."""
    return request.extensions.keys() <= PRICE_NEUTRAL_EXTENSIONS


def cost_of(spec: ModelSpec, decoded: Decoded, *, list_price: bool) -> Money | None:
    """The call's exact cost, or ``None`` when it cannot be known exactly.

    Args:
        spec: The model the call went to.
        decoded: What the provider answered.
        list_price: Whether the catalog's rates apply to the call: it went to
            the provider's public endpoint and no extension changed the price.

    Returns:
        The provider's own figure when it sent one. Otherwise the catalog's
        price for the usage, or ``None`` when the rates do not apply, the model
        has none, the usage was not measured, or it includes a kind of token
        the model has no rate for.
    """
    if decoded.cost is not None:
        return decoded.cost
    if not list_price or spec.rates is None or not decoded.usage.measured:
        return None
    return spec.rates.cost(decoded.usage)


@dataclasses.dataclass(slots=True)
class Call:
    """One model call that has passed its checks and holds its resources.

    Attributes:
        call_id: Shared by every attempt of this call in observer reports.
        spec: The resolved model.
        request: What to send.
        lease: The provider to send it with.
        permit: The rate-limit slot it holds until it finishes.
        retry: The retry policy of the context the call was made in.
        reporter: Delivers attempts to the observers active where the call
            was made.
        list_price: Whether the catalog's rates price this call.
        sleep: Waits between attempts.
        jitter: Returns a uniform random number in [0, 1).
    """

    call_id: str
    spec: ModelSpec
    request: Request
    lease: Lease
    permit: Permit
    retry: RetryPolicy
    reporter: Reporter
    list_price: bool
    sleep: Callable[[float], None]
    jitter: Callable[[], float]
    started: bool = dataclasses.field(default=False, init=False, repr=False)
    finished: bool = dataclasses.field(default=False, init=False, repr=False)

    def run(self) -> Result:
        """Sends the request, retrying as the module docstring says."""
        try:
            return self._send_until_done()
        finally:
            self.finish()

    def open_events(self) -> Iterator[Event]:
        """Returns the call's stream; its resources are freed even if it is never read.

        A started stream frees them when it ends or is closed, after reporting
        any unfinished attempt as cancelled. One dropped before it started has
        no generator frame to do that, so its finalizer frees them instead.
        CPython runs that finalizer before closing a dropped generator, which
        is why it must leave started streams alone.
        """
        events = self._events()
        weakref.finalize(events, self._finish_if_never_started)
        return events

    def finish(self) -> None:
        """Releases the permit and the provider lease. Later calls do nothing."""
        if self.finished:
            return
        self.finished = True
        try:
            self.permit.release()
        finally:
            self.lease.release()

    def _finish_if_never_started(self) -> None:
        if not self.started:
            self.finish()

    def _send_until_done(self) -> Result:
        number = 0
        while True:
            number += 1
            started = time.monotonic()
            try:
                decoded = self.lease.provider.send(self.spec, self.request)
            except ModelError as error:
                delay = retry_delay(self.retry, number, error, self.jitter())
                self._report(number, started, error, will_retry=delay is not None)
                if delay is None:
                    raise
                self.sleep(delay)
                continue
            result = self._result(decoded, started)
            self._report(number, started, result, will_retry=False)
            return result

    def _events(self) -> Iterator[Event]:
        self.started = True
        try:
            number = 0
            completed = False
            while not completed:
                number += 1
                completed = yield from self._stream_attempt(number, time.monotonic())
        finally:
            self.finish()

    def _stream_attempt(self, number: int, started: float) -> Generator[Event, None, bool]:
        """Yields one attempt's events and returns whether the call is complete.

        Returns False only when the attempt failed before its first item and
        the call should try again.
        """
        final: Decoded | None = None
        delivered = False
        try:
            for item in self.lease.provider.stream(self.spec, self.request):
                if isinstance(item, Decoded):
                    final = item
                    break
                delivered = True
                yield item
        except ModelError as error:
            delay = None if delivered else retry_delay(self.retry, number, error, self.jitter())
            self._report(number, started, error, will_retry=delay is not None)
            if delay is None:
                raise
            self.sleep(delay)
            return False
        except GeneratorExit:
            self._report(number, started, self._cancelled(), will_retry=False)
            raise
        if final is None:
            truncated = self._error(
                ErrorKind.TRANSPORT, "the stream ended before its final response"
            )
            self._report(number, started, truncated, will_retry=False)
            raise truncated
        result = self._result(final, started)
        self._report(number, started, result, will_retry=False)
        yield Done(result)
        return True

    def _result(self, decoded: Decoded, started: float) -> Result:
        return Result(
            text=decoded.text,
            reasoning=decoded.reasoning,
            tool_calls=decoded.tool_calls,
            finish_reason=decoded.finish_reason,
            usage=decoded.usage,
            cost=cost_of(self.spec, decoded, list_price=self.list_price),
            model=self.spec.ref,
            request_id=decoded.request_id,
            latency_s=time.monotonic() - started,
            message=decoded.message,
        )

    def _report(
        self, number: int, started: float, outcome: Result | ModelError, *, will_retry: bool
    ) -> None:
        self.permit.record(outcome.kind if isinstance(outcome, ModelError) else None)
        attempt = Attempt(
            call_id=self.call_id,
            number=number,
            model=self.spec.ref,
            latency_s=time.monotonic() - started,
            outcome=outcome,
            will_retry=will_retry,
        )
        self.reporter.report(attempt)

    def _cancelled(self) -> ModelError:
        return self._error(ErrorKind.CANCELLED, "the caller stopped reading the stream")

    def _error(self, kind: ErrorKind, reason: str) -> ModelError:
        ref = self.spec.ref
        return ModelError(
            kind, provider=ref.provider, model=ref.name, reason=reason, before_first_byte=False
        )
