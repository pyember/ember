"""Per-provider concurrency limits that back off when a provider pushes back.

Each provider has a limit on requests in flight. It starts at the cap from the
caller's context (`Context.max_in_flight_per_provider`) and moves by
additive increase, multiplicative decrease:

- A ``RATE_LIMITED`` or ``OVERLOADED`` outcome halves the limit (never below
  one). Outcomes from requests sent before that decrease do not halve it
  again, so one burst of rejections counts as one signal.
- A success adds one to the limit, up to the cap, but only while the provider
  is saturated (in flight >= limit). Serial traffic never raises the limit.

`acquire` returns a future that completes with a `Permit` once the provider
has room. Callers waiting for room sit in an `ember.xcs.runtime.WaitQueue`,
so the scheduler decides who gets each freed slot. The client reads the
future through `ember.xcs.runtime.wait`, so a call waiting for room lets other
work run.
"""

import dataclasses
import threading
from concurrent.futures import Future

from ember.errors import ErrorKind
from ember.xcs import runtime

__all__ = ["Limiter", "LimiterState", "Permit"]

_BACKOFF_KINDS = frozenset({ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED})


@dataclasses.dataclass(frozen=True, slots=True)
class LimiterState:
    """A snapshot of one provider's limiter.

    Attributes:
        limit: Requests allowed in flight.
        in_flight: Permits held.
        waiting: Callers waiting for a permit.
    """

    limit: int
    in_flight: int
    waiting: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Waiter:
    future: "Future[Permit]"
    cap: int


@dataclasses.dataclass(slots=True)
class _Provider:
    """Mutable state for one provider, guarded by the limiter's lock.

    Attributes:
        limit: Requests allowed in flight.
        in_flight: Permits held.
        epoch: Counts halvings; a permit's outcome halves only in its epoch.
        waiters: Callers waiting for room, in the scheduler's order.
        next: The waiter that won the last freed slot but whose own, lower
            cap still holds it back. It keeps its turn.
    """

    limit: int
    in_flight: int = 0
    epoch: int = 0
    waiters: runtime.WaitQueue[_Waiter] = dataclasses.field(default_factory=runtime.WaitQueue)
    next: _Waiter | None = None

    def waiting(self) -> int:
        return len(self.waiters) + (self.next is not None)


class Permit:
    """The right to have one request in flight to one provider.

    Report each attempt's outcome with `record`, then return the slot with
    `release`, exactly once.
    """

    __slots__ = ("_cap", "_epoch", "_limiter", "_provider", "_released")

    def __init__(self, limiter: "Limiter", provider: str, cap: int, epoch: int) -> None:
        self._limiter = limiter
        self._provider = provider
        self._cap = cap
        self._epoch = epoch
        self._released = False

    @property
    def provider(self) -> str:
        """The provider this permit is for."""
        return self._provider

    def record(self, failure: ErrorKind | None) -> None:
        """Reports one attempt's outcome: ``None`` for success, else its error kind.

        Raises:
            RuntimeError: The permit was already released.
        """
        self._limiter._record(self, failure)

    def release(self) -> None:
        """Returns the slot to the provider.

        Raises:
            RuntimeError: The permit was already released.
        """
        self._limiter._release(self)


class Limiter:
    """Adaptive in-flight limits, one per provider name. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._providers: dict[str, _Provider] = {}

    def acquire(self, provider: str, *, cap: int) -> "Future[Permit]":
        """Asks for a permit to send one request to ``provider``.

        Args:
            provider: Provider name.
            cap: The most requests this caller allows in flight to the provider.

        Returns:
            A future that completes with the permit once there is room. Cancel
            it to stop waiting; a cancelled request never takes a slot.

        Raises:
            ValueError: ``cap`` is less than one.
        """
        if cap < 1:
            raise ValueError(f"cap must be at least 1, got {cap}")
        future: Future[Permit] = Future()
        with self._lock:
            state = self._providers.setdefault(provider, _Provider(limit=cap))
            state.waiters.push(_Waiter(future, cap))
            granted = self._admit(provider, state)
        _deliver(granted)
        return future

    def state(self, provider: str) -> LimiterState | None:
        """Returns a snapshot for ``provider``, or ``None`` if it was never used."""
        with self._lock:
            state = self._providers.get(provider)
            if state is None:
                return None
            return LimiterState(state.limit, state.in_flight, state.waiting())

    def _admit(self, provider: str, state: _Provider) -> list[tuple["Future[Permit]", Permit]]:
        granted = []
        while state.in_flight < state.limit and (state.next is not None or state.waiters):
            waiter = state.next if state.next is not None else state.waiters.pop()
            state.next = None
            if waiter.future.cancelled():
                continue
            if state.in_flight >= waiter.cap:
                state.next = waiter
                break
            if not waiter.future.set_running_or_notify_cancel():
                continue
            state.in_flight += 1
            granted.append((waiter.future, Permit(self, provider, waiter.cap, state.epoch)))
        return granted

    def _record(self, permit: Permit, failure: ErrorKind | None) -> None:
        granted = []
        with self._lock:
            _check_held(permit)
            state = self._providers[permit.provider]
            if failure in _BACKOFF_KINDS and permit._epoch == state.epoch:
                state.limit = max(1, state.limit // 2)
                state.epoch += 1
            elif failure is None and state.in_flight >= state.limit:
                state.limit = min(permit._cap, state.limit + 1)
                granted = self._admit(permit.provider, state)
            permit._epoch = state.epoch
        _deliver(granted)

    def _release(self, permit: Permit) -> None:
        with self._lock:
            _check_held(permit)
            permit._released = True
            state = self._providers[permit.provider]
            state.in_flight -= 1
            granted = self._admit(permit.provider, state)
        _deliver(granted)


def _check_held(permit: Permit) -> None:
    if permit._released:
        raise RuntimeError(f"permit for {permit.provider} was already released")


def _deliver(granted: list[tuple["Future[Permit]", Permit]]) -> None:
    # Outside the lock: completing a future runs its callbacks.
    for future, permit in granted:
        future.set_result(permit)
