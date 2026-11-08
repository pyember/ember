"""Per-provider concurrency limits that back off when a provider pushes back.

Each provider has a limit on requests in flight. Every caller also brings a
cap (`Context.max_in_flight_per_provider`) and is admitted only while fewer
than both are in flight. The limit starts at the first caller's cap and moves
by additive increase, multiplicative decrease:

- Until the first decrease, a caller with a higher cap raises the limit to it.
- A ``RATE_LIMITED`` or ``OVERLOADED`` outcome halves the limit (never below
  one). Outcomes from requests sent before that decrease do not halve it
  again, so one burst of rejections counts as one signal.
- A success adds one to the limit, but only while the provider is saturated
  (in flight >= limit) and the limit is below the reporting caller's cap. A
  success never lowers the limit, and serial traffic never raises it.

A caller that would wait while it holds every in-flight slot itself (an open
stream it has not finished reading, outside ``@xcs.jit``) would wait forever;
`acquire` raises `HeldByCaller` instead.

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

__all__ = ["HeldByCaller", "Limiter", "LimiterState", "Permit"]

_BACKOFF_KINDS = frozenset({ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED})


class HeldByCaller(Exception):
    """The caller would wait for a slot that only the caller itself can free."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"every in-flight slot for {provider} is held by this caller")


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
    holder: object | None


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
        held_by: Permits held for a caller that alone can release them.
    """

    limit: int
    in_flight: int = 0
    epoch: int = 0
    waiters: runtime.WaitQueue[_Waiter] = dataclasses.field(default_factory=runtime.WaitQueue)
    next: _Waiter | None = None
    held_by: dict[object, int] = dataclasses.field(default_factory=dict)

    def waiting(self) -> int:
        return len(self.waiters) + (self.next is not None)

    def hold(self, holder: object | None) -> None:
        self.in_flight += 1
        if holder is not None:
            self.held_by[holder] = self.held_by.get(holder, 0) + 1

    def drop(self, holder: object | None) -> None:
        self.in_flight -= 1
        if holder is not None:
            remaining = self.held_by.pop(holder) - 1
            if remaining:
                self.held_by[holder] = remaining


class Permit:
    """The right to have one request in flight to one provider.

    Report each attempt's outcome with `record`, then return the slot with
    `release`, exactly once.
    """

    __slots__ = ("_cap", "_epoch", "_holder", "_limiter", "_provider", "_released")

    def __init__(
        self, limiter: "Limiter", provider: str, cap: int, epoch: int, holder: object | None
    ) -> None:
        self._limiter = limiter
        self._provider = provider
        self._cap = cap
        self._epoch = epoch
        self._holder = holder
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

    def acquire(
        self,
        provider: str,
        *,
        cap: int,
        caller: object | None = None,
        caller_holds: bool = False,
    ) -> "Future[Permit]":
        """Asks for a permit to send one request to ``provider``.

        Args:
            provider: Provider name.
            cap: The most requests this caller allows in flight to the provider.
            caller: Identifies the caller (the client passes its thread), to
                detect a wait that only the caller could end.
            caller_holds: Whether only ``caller`` can release the permit, as
                with a stream it reads itself.

        Returns:
            A future that completes with the permit once there is room. Cancel
            it to stop waiting; a cancelled request never takes a slot.

        Raises:
            ValueError: ``cap`` is less than one.
            HeldByCaller: The caller would wait, and holds every slot in flight.
        """
        if cap < 1:
            raise ValueError(f"cap must be at least 1, got {cap}")
        future: Future[Permit] = Future()
        holder = caller if caller_holds else None
        with self._lock:
            state = self._providers.setdefault(provider, _Provider(limit=cap))
            if state.epoch == 0 and cap > state.limit:
                state.limit = cap
            if state.waiting() == 0 and state.in_flight < min(state.limit, cap):
                state.hold(holder)
                granted = [(future, Permit(self, provider, cap, state.epoch, holder))]
            elif caller is not None and 0 < state.in_flight == state.held_by.get(caller, 0):
                raise HeldByCaller(provider)
            else:
                # Pushed on the caller's greenlet, so the scheduler sees who waits.
                state.waiters.push(_Waiter(future, cap, holder))
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
            state.hold(waiter.holder)
            permit = Permit(self, provider, waiter.cap, state.epoch, waiter.holder)
            granted.append((waiter.future, permit))
        return granted

    def _record(self, permit: Permit, failure: ErrorKind | None) -> None:
        granted = []
        with self._lock:
            _check_held(permit)
            state = self._providers[permit.provider]
            if failure in _BACKOFF_KINDS and permit._epoch == state.epoch:
                state.limit = max(1, state.limit // 2)
                state.epoch += 1
            elif failure is None and state.limit <= state.in_flight and state.limit < permit._cap:
                state.limit += 1
                granted = self._admit(permit.provider, state)
            permit._epoch = state.epoch
        _deliver(granted)

    def _release(self, permit: Permit) -> None:
        with self._lock:
            _check_held(permit)
            permit._released = True
            state = self._providers[permit.provider]
            state.drop(permit._holder)
            granted = self._admit(permit.provider, state)
        _deliver(granted)


def _check_held(permit: Permit) -> None:
    if permit._released:
        raise RuntimeError(f"permit for {permit.provider} was already released")


def _deliver(granted: list[tuple["Future[Permit]", Permit]]) -> None:
    # Outside the lock: completing a future runs its callbacks.
    for future, permit in granted:
        future.set_result(permit)
