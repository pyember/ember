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
- never once a stream has delivered its first item, which `ModelError`
  marks by ``before_first_byte=False`` and so ``retryable=False``.

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
from ember.llm.observe import Attempt, Observer
from ember.llm.pool import Lease
from ember.llm.provider import Decoded
from ember.llm.request import Request
from ember.llm.result import Done, Event, Money, Result
from ember.llm.spec import ModelSpec

__all__ = ["Call", "cost_of", "retry_delay"]

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


def cost_of(spec: ModelSpec, decoded: Decoded) -> Money | None:
    """The call's cost: the provider's own figure if it sent one, else the catalog's."""
    if decoded.cost is not None:
        return decoded.cost
    if spec.rates is None or not decoded.usage.measured:
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
        observers: The observers active where the call was made.
        sleep: Waits between attempts.
        jitter: Returns a uniform random number in [0, 1).
    """

    call_id: str
    spec: ModelSpec
    request: Request
    lease: Lease
    permit: Permit
    retry: RetryPolicy
    observers: tuple[Observer, ...]
    sleep: Callable[[float], None]
    jitter: Callable[[], float]
    finished: bool = dataclasses.field(default=False, init=False, repr=False)

    def run(self) -> Result:
        """Sends the request, retrying as the module docstring says."""
        try:
            return self._send_until_done()
        finally:
            self.finish()

    def open_events(self) -> Iterator[Event]:
        """Returns the call's stream; its resources are freed even if it is never read."""
        events = self._events()
        weakref.finalize(events, self.finish)
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
        try:
            for item in self.lease.provider.stream(self.spec, self.request):
                if isinstance(item, Decoded):
                    final = item
                    break
                yield item
        except ModelError as error:
            delay = retry_delay(self.retry, number, error, self.jitter())
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
            cost=cost_of(self.spec, decoded),
            model=self.spec.ref,
            request_id=decoded.request_id,
            latency_s=time.monotonic() - started,
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
        for observer in self.observers:
            observer.on_attempt(attempt)

    def _cancelled(self) -> ModelError:
        return self._error(ErrorKind.CANCELLED, "the caller stopped reading the stream")

    def _error(self, kind: ErrorKind, reason: str) -> ModelError:
        ref = self.spec.ref
        return ModelError(
            kind, provider=ref.provider, model=ref.name, reason=reason, before_first_byte=False
        )
