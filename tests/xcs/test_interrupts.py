"""Interrupts inside a scope, and workers that cannot start.

Each in-process scenario runs on a helper thread (a scope of its own) with a
bounded join, so a regression fails fast instead of hanging the run. The real
SIGINT scenarios run in a subprocess, where the signal reaches its main
thread without touching pytest's own.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
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


def call(seconds: float) -> float:
    return runtime.wait(runtime.dispatch(lambda: (time.sleep(seconds), seconds)[1], label="m"))


def test_an_interrupt_from_an_item_after_its_wait_leaves_the_scope_usable() -> None:
    def step(i: int) -> int:
        call(0.05)
        if i == 0:
            raise KeyboardInterrupt
        return i

    @xcs.jit
    def agent() -> list[object]:
        try:
            xcs.map(step, range(2))
        except KeyboardInterrupt:
            caught = True
        direct = call(0.01)
        return [caught, direct, xcs.map(call, [0.01, 0.02])]

    assert on_own_thread(agent) == [True, 0.01, [0.01, 0.02]]


def test_an_interrupt_before_an_items_first_wait_stops_the_items_already_started() -> None:
    log: list[str] = []

    def item(i: int) -> None:
        try:
            if i == 1:
                raise KeyboardInterrupt
            call(0.05)
            log.append(f"item {i} resumed and sent another request")
            runtime.dispatch(lambda: log.append("late request sent"), label="late")
            raise ValueError(f"item {i} failed after its map raised")
        finally:
            log.append(f"item {i} unwound")

    @xcs.jit
    def agent() -> str:
        try:
            xcs.map(item, range(2))
        except KeyboardInterrupt:
            log.append("map raised")
        return "summary"

    assert on_own_thread(agent) == "summary"
    time.sleep(0.1)
    assert log == ["item 1 unwound", "item 0 unwound", "map raised"]


def test_an_item_stopped_by_an_interrupt_still_has_its_requests_waited_for() -> None:
    finished = threading.Event()

    def item(i: int) -> None:
        if i == 1:
            raise KeyboardInterrupt
        runtime.dispatch(lambda: (time.sleep(0.1), finished.set()), label="sent before stop")
        call(1.0)

    @xcs.jit
    def agent() -> bool:
        try:
            xcs.map(item, range(2))
        except KeyboardInterrupt:
            pass
        return finished.is_set()

    assert on_own_thread(agent) is False
    assert finished.is_set()


def test_other_items_stay_held_while_a_stopped_item_waits_in_its_finally() -> None:
    log: list[str] = []

    def item(i: int) -> None:
        if i == 0:
            try:
                call(0.5)
            finally:
                log.append(f"item 0 cleanup read {call(0.2)}")
        elif i == 1:
            call(0.05)
            raise KeyboardInterrupt
        else:
            call(0.1)
            log.append("item 2 continued past its wait")
            runtime.dispatch(lambda: log.append("item 2 sent a request"), label="late")
            call(1.0)

    @xcs.jit
    def program() -> None:
        try:
            xcs.map(item, range(3))
        except KeyboardInterrupt:
            log.append("map raised")

    on_own_thread(program)
    time.sleep(0.1)
    assert log == ["item 0 cleanup read 0.2", "map raised"]


def test_what_a_held_item_started_stays_held_too() -> None:
    log: list[str] = []

    def unwinding_leaf(_: int) -> None:
        try:
            call(0.5)
        finally:
            log.append(f"branch 0 leaf cleanup read {call(0.2)}")

    def running_leaf(_: int) -> None:
        call(0.1)
        log.append("branch 2 leaf continued past its wait")

    def branch(i: int) -> None:
        if i == 0:
            xcs.map(unwinding_leaf, range(1))
        elif i == 1:
            call(0.05)
            raise KeyboardInterrupt
        else:
            xcs.map(running_leaf, range(1))

    @xcs.jit
    def program() -> None:
        try:
            xcs.map(branch, range(3))
        except KeyboardInterrupt:
            log.append("map raised")

    on_own_thread(program)
    time.sleep(0.1)
    assert log == ["branch 0 leaf cleanup read 0.2", "map raised"]


def run_script(script: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    env = {**os.environ, "HOME": str(tmp_path), "EMBER_CONFIG": str(tmp_path / "none.yaml")}
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result, time.perf_counter() - started


SIGINT_WHILE_PARKED = """
    import os, signal, threading, time
    from ember import xcs
    from ember.xcs import runtime

    def call(seconds):
        return runtime.wait(runtime.dispatch(lambda: (time.sleep(seconds), seconds)[1], label="m"))

    @xcs.jit
    def agent():
        try:
            xcs.map(call, [2.0, 2.0])
        except KeyboardInterrupt:
            print("interrupted", flush=True)
        return xcs.map(call, [0.01, 0.02])

    threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
    print(agent(), flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_sigint_while_parked_is_raised_in_the_body_and_the_scope_stays_usable(
    tmp_path: Path,
) -> None:
    result, _ = run_script(SIGINT_WHILE_PARKED, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "interrupted\n[0.01, 0.02]\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_an_uncaught_sigint_ends_the_jit_call_without_waiting_for_requests(
    tmp_path: Path,
) -> None:
    script = SIGINT_WHILE_PARKED.replace("except KeyboardInterrupt:", "except ValueError:")
    result, elapsed_s = run_script(script, tmp_path)
    assert result.returncode != 0
    assert "KeyboardInterrupt" in result.stderr
    assert elapsed_s < 2.0


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
