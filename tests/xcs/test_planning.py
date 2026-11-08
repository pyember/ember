"""Critical-path planning: who gets a freed slot when requests wait for capacity.

The limiter is `ember.llm.limiter.Limiter`, which keeps its waiters in a
`runtime.WaitQueue`; the calls take and release permits as the client does.
"""

import threading
import time
from collections.abc import Callable

import pytest

from ember import xcs
from ember.llm.limiter import Limiter
from ember.xcs import runtime


def limited_call(capacity: int, latency_s: Callable[[str], float]) -> Callable[[str], str]:
    """A model call shaped like the client's, through ember's own limiter.

    It waits for a permit with `runtime.wait`, dispatches, and releases the
    permit inside the task, so a freed slot is granted before the reader
    wakes. No outcome is recorded, so the limit stays at ``capacity``.
    """
    limiter = Limiter()

    def call(prompt: str) -> str:
        permit = runtime.wait(limiter.acquire("fake", cap=capacity))

        def task() -> str:
            try:
                time.sleep(latency_s(prompt))
                return prompt
            finally:
                permit.release()

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


def test_a_limiter_using_the_queue_grants_slots_by_predicted_remaining_path() -> None:
    durations = {"first": 0.02, "short": 0.01, "long": 0.08, "middle": 0.04}
    call = limited_call(1, durations.__getitem__)
    granted: list[str] = []

    def item(name: str) -> None:
        call(name)
        granted.append(name)

    @xcs.jit
    def program() -> None:
        xcs.map(item, list(durations))

    program()
    assert granted == ["first", "short", "long", "middle"]
    granted.clear()
    program()
    assert granted == ["first", "long", "middle", "short"]


@pytest.mark.slow
def test_planning_shortens_the_makespan_under_a_capacity_limit() -> None:
    latency_s = 0.15
    call = limited_call(2, lambda _: latency_s)
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
    # FIFO runs the four rounds of singles first and the chain after: 8 rounds.
    # With the plan the chain starts in round 2. The lower bound is 6 rounds,
    # but the client frees a permit inside the task, before the chain's next
    # request can queue, so a slot freed by a link may go to a single first:
    # the worst case is 7 rounds. Half a round of margin absorbs timer noise.
    assert unplanned >= 8 * latency_s
    assert planned <= unplanned - 0.5 * latency_s, f"{unplanned=:.3f}s {planned=:.3f}s"
