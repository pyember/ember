"""Critical-path planning: who gets a freed slot when requests wait for capacity."""

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

import pytest

from ember import xcs
from ember.xcs import runtime


class Limiter:
    """A fixed-capacity limiter that keeps its waiters in a `runtime.WaitQueue`.

    It releases inside the dispatched task, as the client does, so a freed slot
    is granted before the task's reader wakes.
    """

    def __init__(self, capacity: int) -> None:
        self._lock = threading.Lock()
        self._free = capacity
        self._waiters: runtime.WaitQueue[Future[None]] = runtime.WaitQueue()

    def acquire(self) -> "Future[None]":
        permit: Future[None] = Future()
        with self._lock:
            granted = self._free > 0
            if granted:
                self._free -= 1
            else:
                self._waiters.push(permit)
        if granted:
            permit.set_result(None)
        return permit

    def release(self) -> None:
        with self._lock:
            if not self._waiters:
                self._free += 1
                return
            permit = self._waiters.pop()
        permit.set_result(None)


def limited_call(limiter: Limiter, latency_s: float) -> Callable[[str], str]:
    def call(prompt: str) -> str:
        runtime.wait(limiter.acquire())

        def task() -> str:
            try:
                time.sleep(latency_s)
                return prompt
            finally:
                limiter.release()

        return runtime.wait(runtime.dispatch(task, label="limited"))

    return call


def test_outside_jit_the_queue_is_first_in_first_out() -> None:
    queue: runtime.WaitQueue[str] = runtime.WaitQueue()
    for item in "abc":
        queue.push(item)
    assert [queue.pop() for _ in range(3)] == ["a", "b", "c"]
    assert len(queue) == 0


def test_popping_an_empty_queue_raises() -> None:
    with pytest.raises(IndexError):
        runtime.WaitQueue[str]().pop()


def recorded_durations(queue: "runtime.WaitQueue[int]", durations: list[float]) -> xcs.Jitted:
    """A program whose items push to ``queue`` and then make one call of a known duration."""

    def item(i: int) -> None:
        queue.push(i)
        runtime.wait(runtime.dispatch(lambda: time.sleep(durations[i]), label="sleep"))

    @xcs.jit
    def program(label: str) -> None:
        runtime.wait(runtime.dispatch(lambda: None, label=label))
        xcs.map(item, range(len(durations)))

    return program


def test_within_a_jit_call_the_longest_predicted_path_goes_first() -> None:
    queue: runtime.WaitQueue[int] = runtime.WaitQueue()
    program = recorded_durations(queue, [0.01, 0.08, 0.04])
    program("same")
    assert [queue.pop() for _ in range(3)] == [0, 1, 2]
    program("same")
    assert [queue.pop() for _ in range(3)] == [1, 2, 0]


def test_a_run_that_departs_from_the_plan_drops_it() -> None:
    queue: runtime.WaitQueue[int] = runtime.WaitQueue()
    program = recorded_durations(queue, [0.01, 0.08, 0.04])
    program("first label")
    [queue.pop() for _ in range(3)]
    program("another label")
    assert [queue.pop() for _ in range(3)] == [0, 1, 2]


def test_jit_calls_get_slots_in_arrival_order() -> None:
    queue: runtime.WaitQueue[str] = runtime.WaitQueue()
    durations = {"short": 0.01, "long": 0.08}

    def item(name: str) -> None:
        queue.push(name)
        if name == "short":
            outsider = threading.Thread(target=queue.push, args=("outsider",))
            outsider.start()
            outsider.join()
        runtime.wait(runtime.dispatch(lambda: time.sleep(durations[name]), label="sleep"))

    @xcs.jit
    def program() -> None:
        xcs.map(item, ["short", "long"])

    program()
    [queue.pop() for _ in range(3)]
    program()
    assert [queue.pop() for _ in range(3)] == ["long", "outsider", "short"]


@pytest.mark.slow
def test_planning_shortens_the_makespan_under_a_capacity_limit() -> None:
    latency_s = 0.15
    call = limited_call(Limiter(capacity=2), latency_s)
    singles, chain = 8, 4

    def item(i: int) -> str:
        if i < singles:
            return call(f"single {i}")
        text = "chain"
        for step in range(chain):
            text = call(f"{text}>{step}")
        return text

    @xcs.jit
    def program() -> list[str]:
        return xcs.map(item, range(singles + 1))

    makespans = []
    for _ in range(2):
        started = time.perf_counter()
        program()
        makespans.append(time.perf_counter() - started)
    unplanned, planned = makespans
    # FIFO runs the singles first and the chain last: 8 rounds of latency.
    # The plan starts the chain first: close to max(chain, 12 calls / 2) = 6.
    assert unplanned >= 8 * latency_s
    assert planned <= unplanned - 1.0 * latency_s, f"{unplanned=:.3f}s {planned=:.3f}s"
