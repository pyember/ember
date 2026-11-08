"""The per-provider AIMD limiter."""

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

import pytest

from ember.errors import ErrorKind
from ember.llm.limiter import Limiter, LimiterState, Permit


def _grant(limiter: Limiter, provider: str = "openai", cap: int = 8) -> Permit:
    future = limiter.acquire(provider, cap=cap)
    assert future.done(), "expected room for a permit"
    return future.result()


def _hold(limiter: Limiter, count: int, cap: int) -> list[Permit]:
    return [_grant(limiter, cap=cap) for _ in range(count)]


def test_limit_starts_at_the_cap() -> None:
    limiter = Limiter()
    permit = _grant(limiter, cap=8)
    assert limiter.state("openai") == LimiterState(limit=8, in_flight=1, waiting=0)
    permit.release()
    assert limiter.state("openai") == LimiterState(limit=8, in_flight=0, waiting=0)


def test_serial_successes_never_raise_the_limit() -> None:
    limiter = Limiter()
    first = _grant(limiter, cap=64)
    first.record(ErrorKind.RATE_LIMITED)
    first.release()
    assert limiter.state("openai") == LimiterState(limit=32, in_flight=0, waiting=0)
    for _ in range(2_000):
        permit = _grant(limiter, cap=64)
        permit.record(None)
        permit.release()
    assert limiter.state("openai") == LimiterState(limit=32, in_flight=0, waiting=0)


def test_successes_grow_the_limit_only_while_saturated() -> None:
    limiter = Limiter()
    first = _grant(limiter, cap=4)
    first.record(ErrorKind.OVERLOADED)
    first.release()
    held = _hold(limiter, 2, cap=4)  # limit 2: saturated
    held[0].record(None)
    assert limiter.state("openai") == LimiterState(limit=3, in_flight=2, waiting=0)
    held[0].release()
    held[1].record(None)  # one in flight under a limit of three: not saturated
    assert limiter.state("openai") == LimiterState(limit=3, in_flight=1, waiting=0)


def test_growth_stops_at_the_cap() -> None:
    limiter = Limiter()
    held = _hold(limiter, 2, cap=2)
    for permit in held:
        permit.record(None)
    assert limiter.state("openai") == LimiterState(limit=2, in_flight=2, waiting=0)


def test_one_burst_of_rejections_halves_the_limit_once() -> None:
    limiter = Limiter()
    burst = _hold(limiter, 8, cap=8)
    for permit in burst:
        permit.record(ErrorKind.RATE_LIMITED)
    assert limiter.state("openai") == LimiterState(limit=4, in_flight=8, waiting=0)
    burst[0].record(ErrorKind.RATE_LIMITED)  # a retry sent after the decrease counts
    assert limiter.state("openai") == LimiterState(limit=2, in_flight=8, waiting=0)


@pytest.mark.parametrize(
    "kind",
    [
        ErrorKind.SERVER,
        ErrorKind.TRANSPORT,
        ErrorKind.AUTH,
        ErrorKind.QUOTA,
        ErrorKind.MALFORMED,
        ErrorKind.CANCELLED,
    ],
)
def test_other_failures_leave_the_limit_alone(kind: ErrorKind) -> None:
    limiter = Limiter()
    permit = _grant(limiter, cap=8)
    permit.record(kind)
    assert limiter.state("openai") == LimiterState(limit=8, in_flight=1, waiting=0)


def test_the_limit_never_drops_below_one() -> None:
    limiter = Limiter()
    permit = _grant(limiter, cap=1)
    for _ in range(5):
        permit.record(ErrorKind.RATE_LIMITED)
    assert limiter.state("openai") == LimiterState(limit=1, in_flight=1, waiting=0)


def test_a_permit_releases_exactly_once() -> None:
    limiter = Limiter()
    permit = _grant(limiter)
    permit.release()
    with pytest.raises(RuntimeError, match="already released"):
        permit.release()
    with pytest.raises(RuntimeError, match="already released"):
        permit.record(None)
    assert limiter.state("openai") == LimiterState(limit=8, in_flight=0, waiting=0)


def test_without_a_scheduler_waiters_get_freed_slots_in_arrival_order() -> None:
    limiter = Limiter()
    first = _grant(limiter, cap=1)
    second: Future[Permit] = limiter.acquire("openai", cap=1)
    third: Future[Permit] = limiter.acquire("openai", cap=1)
    assert not second.done() and not third.done()
    first.release()
    assert second.done() and not third.done()
    second.result().release()
    assert third.done()
    assert limiter.state("openai") == LimiterState(limit=1, in_flight=1, waiting=0)


def test_a_cancelled_waiter_never_takes_a_slot() -> None:
    limiter = Limiter()
    first = _grant(limiter, cap=1)
    waiting = limiter.acquire("openai", cap=1)
    assert waiting.cancel()
    first.release()
    assert limiter.state("openai") == LimiterState(limit=1, in_flight=0, waiting=0)


def test_a_lower_cap_bounds_its_own_request() -> None:
    limiter = Limiter()
    _grant(limiter, cap=4)
    assert not limiter.acquire("openai", cap=1).done()


def test_providers_do_not_share_capacity() -> None:
    limiter = Limiter()
    _grant(limiter, "openai", cap=1)
    assert not limiter.acquire("openai", cap=1).done()
    assert limiter.acquire("anthropic", cap=1).done()


def _wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.001)


def test_threads_block_at_the_cap_and_all_finish() -> None:
    limiter = Limiter()
    lock = threading.Lock()
    release = threading.Event()
    held = peak = 0

    def worker() -> None:
        nonlocal held, peak
        permit = limiter.acquire("openai", cap=3).result(timeout=10)
        with lock:
            held += 1
            peak = max(peak, held)
        release.wait(timeout=10)
        with lock:
            held -= 1
        permit.release()

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for thread in threads:
        thread.start()
    _wait_for(lambda: limiter.state("openai") == LimiterState(limit=3, in_flight=3, waiting=29))
    assert peak == 3
    release.set()
    for thread in threads:
        thread.join(timeout=10)
    assert peak == 3
    assert limiter.state("openai") == LimiterState(limit=3, in_flight=0, waiting=0)


def test_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Limiter().acquire("openai", cap=0)


def test_a_newcomer_never_overtakes_a_waiting_caller() -> None:
    limiter = Limiter()
    held = _grant(limiter, cap=4)
    narrow = limiter.acquire("openai", cap=1)  # waits: its own cap of one is in use
    wide = limiter.acquire("openai", cap=4)  # room under four, but arrived later
    assert not narrow.done() and not wide.done()
    held.release()
    assert narrow.done() and wide.done()
    assert limiter.state("openai") == LimiterState(limit=4, in_flight=2, waiting=0)


def test_a_cancelled_head_does_not_hold_back_a_newcomer() -> None:
    limiter = Limiter()
    held = _grant(limiter, cap=1)
    abandoned = limiter.acquire("openai", cap=1)
    assert abandoned.cancel()
    held.release()
    assert limiter.acquire("openai", cap=1).done()
    assert limiter.state("openai") == LimiterState(limit=1, in_flight=1, waiting=0)
