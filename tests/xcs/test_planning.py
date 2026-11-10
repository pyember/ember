"""Critical-path planning: who gets a freed slot when requests wait for capacity.

The limiter is `ember.llm.limiter.Limiter`, which keeps its waiters in a
`runtime.WaitQueue`; the calls take and release permits as the client does.

A plan is what the previous run measured, and a loaded machine can stretch any
measurement. So these tests read the order a plan implies from the recorded
graph, and rely on one comparison only: a call that sleeps took longer than a
call that does not.
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


def recorded_remaining_s(program: xcs.Jitted) -> tuple[float, ...]:
    """Each call's remaining path in the program's last run, in dispatch order."""
    graph = program.graph()
    assert graph is not None
    return graph.remaining_s()


def recorded_durations(queue: "runtime.WaitQueue[int]", durations: list[float]) -> xcs.Jitted:
    """A program whose items push to ``queue`` and then make one call that sleeps its duration."""

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
    program = recorded_durations(queue, [0.0, 0.08, 0.04])
    program("same")
    assert [queue.pop() for _ in range(3)] == [0, 1, 2]
    _, *recorded = recorded_remaining_s(program)  # The labelled call, then the items.
    assert recorded[1] > recorded[0], recorded
    program("same")
    longest_first = sorted(range(3), key=lambda i: -recorded[i])
    assert [queue.pop() for _ in range(3)] == longest_first, recorded


def test_a_run_that_departs_from_the_plan_drops_it() -> None:
    queue: runtime.WaitQueue[int] = runtime.WaitQueue()
    program = recorded_durations(queue, [0.01, 0.08, 0.04])
    program("first label")
    [queue.pop() for _ in range(3)]
    program("another label")
    assert [queue.pop() for _ in range(3)] == [0, 1, 2]


def test_jit_calls_get_slots_in_arrival_order() -> None:
    queue: runtime.WaitQueue[str] = runtime.WaitQueue()
    durations = {"short": 0.0, "long": 0.08}

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
    short, long = recorded_remaining_s(program)
    assert long > short, (short, long)
    program()
    assert [queue.pop() for _ in range(3)] == ["long", "outsider", "short"]


def test_a_limiter_using_the_queue_grants_slots_by_predicted_remaining_path() -> None:
    # "first" takes the only slot on arrival and holds it while the others queue.
    durations = {"first": 0.08, "short": 0.0, "long": 0.08, "middle": 0.04}
    granted: list[str] = []

    def latency_s(name: str) -> float:
        granted.append(name)  # Called by the task, while it holds the slot.
        return durations[name]

    call = limited_call(1, latency_s)

    @xcs.jit
    def program() -> None:
        xcs.map(call, list(durations))

    program()
    assert granted == ["first", "short", "long", "middle"]
    recorded = dict(zip(granted, recorded_remaining_s(program), strict=True))
    assert recorded["long"] > recorded["short"], recorded
    granted.clear()
    program()
    longest_first = sorted(["short", "long", "middle"], key=lambda name: -recorded[name])
    assert granted == ["first", *longest_first], recorded


@pytest.mark.slow
def test_planning_starts_the_critical_path_first_under_a_capacity_limit() -> None:
    # Eight single calls and one chain of four dependent calls share two slots.
    # Without a plan the chain's first call waits behind every single; with the
    # plan from the first run, it takes the first slot that frees. The order is
    # the scheduler's decision, read from the recorded graph, so a slow machine
    # cannot change it. The latency only has to keep one single shorter than the
    # whole chain in the first run's measurements.
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

    def calls_before_the_chain() -> int:
        graph = program.graph()
        assert graph is not None
        second_link = next(node for node in graph.nodes if node.deps)
        [first_link] = second_link.deps
        return first_link  # A node's id is its position in dispatch order.

    program()
    assert calls_before_the_chain() == singles
    program()
    assert calls_before_the_chain() == 2
