"""The threads that run model requests' network I/O.

One process-wide pool of named daemon threads runs every task given to
`ember.xcs.runtime.dispatch` and `dispatch_stream`. Tasks are leaf I/O that
never wait on other ember work.

The pool never makes a task wait for a thread: when no thread is idle it
starts one. Concurrency is bounded by the providers' permits, which every
task holds for its whole run (retry backoff included), so the pool must not
be a tighter bound: a provider whose permits are all held in backoff would
otherwise delay requests to every other provider. Threads exit after
`IDLE_EXIT_S` without work, so an idle process holds none. A forked child
starts with a fresh, empty pool, because the parent's threads do not exist in
it.
"""

import collections
import itertools
import os
import sys
import threading
from collections.abc import Callable

__all__ = ["IDLE_EXIT_S", "WorkerPool", "submit"]

#: Seconds a worker waits for a new task before its thread exits.
IDLE_EXIT_S = 30.0


class WorkerPool:
    """Threads that start a task as soon as it is submitted.

    Args:
        idle_exit_s: Seconds an idle thread waits for work before exiting.
        name: Prefix of the threads' names.
    """

    def __init__(self, idle_exit_s: float, name: str) -> None:
        self._idle_exit_s = idle_exit_s
        self._name = name
        self._lock = threading.Lock()
        self._has_work = threading.Condition(self._lock)
        self._tasks: collections.deque[Callable[[], None]] = collections.deque()
        self._threads = 0
        self._idle = 0
        self._serial = itertools.count(1)

    def submit(self, task: Callable[[], None]) -> None:
        """Runs ``task`` on an idle thread, or on a new thread started for it.

        All or nothing: when a new thread is needed and cannot start (the
        process is at its thread limit), the error is raised here and ``task``
        never runs. ``task`` must not raise: it reports its own outcome, for
        example by completing a future.

        Raises:
            RuntimeError: No thread is idle and a new one could not start.
        """
        with self._lock:
            if self._idle > len(self._tasks):
                self._tasks.append(task)
                self._has_work.notify()
                return
            thread = threading.Thread(
                target=self._work,
                args=(task,),
                name=f"{self._name}-{next(self._serial)}",
                daemon=True,
            )
            thread.start()
            self._threads += 1

    def thread_count(self) -> int:
        """Threads currently alive in the pool."""
        with self._lock:
            return self._threads

    def _work(self, first: Callable[[], None]) -> None:
        task: Callable[[], None] | None = first
        try:
            while task is not None:
                task()
                task = self._next_task()
        except BaseException:
            with self._lock:
                self._threads -= 1
            raise

    def _next_task(self) -> Callable[[], None] | None:
        with self._lock:
            while not self._tasks:
                self._idle += 1
                try:
                    woken = self._has_work.wait(timeout=self._idle_exit_s)
                finally:
                    self._idle -= 1
                if not woken and not self._tasks:
                    self._threads -= 1
                    return None
            return self._tasks.popleft()


_pool = WorkerPool(IDLE_EXIT_S, name="ember-xcs-worker")


def submit(task: Callable[[], None]) -> None:
    """Queues ``task`` on the process-wide pool; see `WorkerPool.submit`."""
    _pool.submit(task)


def _reset_after_fork() -> None:
    global _pool
    _pool = WorkerPool(IDLE_EXIT_S, name="ember-xcs-worker")


if sys.platform != "win32":
    os.register_at_fork(after_in_child=_reset_after_fork)
