"""The worker pool: never a tighter bound than the permits, owned, fork-safe."""

import os
import signal
import sys
import threading
import time
import warnings

import pytest

from ember import xcs
from ember.xcs import runtime, workers


def test_a_provider_holding_threads_in_backoff_does_not_delay_another() -> None:
    pool = workers.WorkerPool(idle_exit_s=1.0, name="test-pool")
    backoff_over = threading.Event()
    other_done = threading.Event()
    for _ in range(300):
        pool.submit(lambda: backoff_over.wait())
    started = time.perf_counter()
    pool.submit(other_done.set)
    try:
        assert other_done.wait(2.0)
        assert time.perf_counter() - started < 1.0
    finally:
        backoff_over.set()


def test_idle_threads_exit() -> None:
    pool = workers.WorkerPool(idle_exit_s=0.05, name="test-pool")
    ran = threading.Event()
    pool.submit(ran.set)
    assert ran.wait(1.0)
    deadline = time.monotonic() + 2.0
    while pool.thread_count() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.thread_count() == 0


def test_an_idle_thread_is_reused_rather_than_a_new_one_started() -> None:
    pool = workers.WorkerPool(idle_exit_s=1.0, name="test-pool")
    names: list[str] = []
    for _ in range(3):
        done = threading.Event()
        pool.submit(lambda done=done: (names.append(threading.current_thread().name), done.set()))
        assert done.wait(1.0)
        time.sleep(0.01)
    assert pool.thread_count() == 1
    assert len(set(names)) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="os.fork is POSIX only")
def test_a_forked_child_gets_a_fresh_pool() -> None:
    @xcs.jit
    def program() -> int:
        return runtime.wait(runtime.dispatch(os.getpid, label="pid"))

    assert program() == os.getpid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        child = os.fork()
    if child == 0:
        signal.alarm(10)
        status = 0 if program() == os.getpid() else 1
        os._exit(status)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.skipif(sys.platform == "win32", reason="os.fork is POSIX only")
def test_a_child_forked_inside_a_scope_fails_at_its_first_wait() -> None:
    @xcs.jit
    def program() -> int:
        pending = runtime.dispatch(lambda: (time.sleep(0.2), 7)[1], label="in flight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            child = os.fork()
        if child == 0:
            signal.alarm(10)
            try:
                runtime.wait(pending)
            except RuntimeError:
                os._exit(0)
            os._exit(1)
        assert runtime.wait(pending) == 7
        return child

    child = program()
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
