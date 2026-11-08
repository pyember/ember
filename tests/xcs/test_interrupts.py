"""Workers that cannot start.

Each in-process scenario runs on a helper thread (a scope of its own) with a
bounded join, so a regression fails fast instead of hanging the run.
"""

import threading
import time
from collections.abc import Callable
from typing import TypeVar

import pytest

from ember import xcs
from ember.xcs import runtime, workers

T = TypeVar("T")


def on_own_thread(scenario: Callable[[], T], within_s: float = 10.0) -> T:
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = scenario()
        except BaseException as error:  # Reported to the test below.
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(within_s)
    assert not thread.is_alive(), f"the scenario did not finish within {within_s}s"
    if "error" in outcome:
        raise AssertionError(f"the scenario raised {outcome['error']!r}")
    return outcome["result"]  # type: ignore[return-value]


def test_a_task_whose_thread_cannot_start_never_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = workers.WorkerPool(idle_exit_s=1.0, name="test-start-failure")
    ran: list[str] = []
    real_start = threading.Thread.start
    failures = [RuntimeError("can't start new thread")]

    def start_or_fail(thread: threading.Thread) -> None:
        if failures:
            raise failures.pop()
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_or_fail)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        pool.submit(lambda: ran.append("failed submit"))
    assert pool.thread_count() == 0
    done = threading.Event()
    pool.submit(lambda: (ran.append("next submit"), done.set()))
    assert done.wait(2.0)
    time.sleep(0.05)
    assert ran == ["next submit"]
    live = [t for t in threading.enumerate() if t.name.startswith("test-start-failure")]
    assert pool.thread_count() == len(live) == 1


def test_a_dispatch_that_cannot_start_raises_at_the_call_and_leaves_nothing_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_submit = workers.submit
    failures = [RuntimeError("can't start new thread")]

    def submit_or_fail(task: Callable[[], None]) -> None:
        if failures:
            raise failures.pop()
        real_submit(task)

    monkeypatch.setattr(workers, "submit", submit_or_fail)
    sent: list[str] = []

    @xcs.jit
    def agent() -> str:
        with pytest.raises(RuntimeError):
            runtime.dispatch(lambda: sent.append("never"), label="refused")
        return runtime.wait(runtime.dispatch(lambda: "sent", label="accepted"))

    assert on_own_thread(agent, within_s=5.0) == "sent"
    graph = agent.graph()
    assert graph is not None
    assert [node.label for node in graph.nodes] == ["accepted"]
    assert sent == []
