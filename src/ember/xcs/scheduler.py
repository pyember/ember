"""Cooperative scheduling of greenlets on one thread.

Inside `jit` and `map`, user code runs in greenlets on the caller's thread. A
greenlet that needs a pending result parks until the result is complete, and
the scheduler runs another greenlet meanwhile. One greenlet runs at a time, so
user code never runs in parallel with other user code. Worker threads complete
futures and so wake parked greenlets through a thread-safe queue; they never
run user code.

A scheduler serves one scope, the outermost `jit` or `map` call on a thread.
The greenlet that opened the scope (the owner) keeps running the scope's body.
A hub greenlet, whose parent is the owner, picks the next greenlet to run each
time the running one parks.

Invariant: every suspended greenlet is woken exactly once. A parked greenlet is
queued once by its future's callback, and a child started by `start` returns
straight to the greenlet that started it the first time it parks.
"""

import contextvars
import queue
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar

import greenlet

__all__ = ["Scheduler"]

T = TypeVar("T")


class Scheduler:
    """Runs greenlets on the thread that created it, one at a time.

    Create it on the greenlet that will own the scope, and call `close` on the
    same greenlet when the scope ends.
    """

    def __init__(self) -> None:
        self._owner = greenlet.getcurrent()
        self._ready: queue.SimpleQueue[greenlet.greenlet] = queue.SimpleQueue()
        self._hub: greenlet.greenlet | None = None
        self._children: set[greenlet.greenlet] = set()
        self._starting: set[greenlet.greenlet] = set()
        self._closing = False

    def park(self, future: "Future[T]") -> None:
        """Suspends the running greenlet until ``future`` is done.

        Other greenlets run meanwhile. Returns at once if ``future`` is done.
        """
        if future.done():
            return
        current = greenlet.getcurrent()
        future.add_done_callback(lambda _: self._ready.put(current))
        self._suspend(current)

    def start(self, run: Callable[[], None], context: contextvars.Context) -> None:
        """Runs ``run`` in a new greenlet until it first parks or finishes.

        Control then returns to the caller; the new greenlet resumes later,
        when what it parked on is done. ``run`` executes in ``context``.
        ``run`` must catch the exceptions it means to report; anything it lets
        escape propagates to the caller of `start` or to the scope's owner.
        """
        spawner = greenlet.getcurrent()
        child = greenlet.greenlet(self._child_main(run), parent=spawner)
        child.gr_context = context
        self._children.add(child)
        self._starting.add(child)
        child.switch()

    def close(self) -> None:
        """Ends the scope, on the owner, after its body has finished.

        Greenlets still suspended (possible only after an interrupt such as
        KeyboardInterrupt) are killed with `greenlet.GreenletExit`.
        """
        for child in list(self._children):
            child.parent = self._owner
            child.throw(greenlet.GreenletExit)
        self._children.clear()
        if self._hub is not None and not self._hub.dead:
            self._closing = True
            self._hub.switch()

    def _child_main(self, run: Callable[[], None]) -> Callable[[], None]:
        def main() -> None:
            current = greenlet.getcurrent()
            try:
                run()
            finally:
                self._children.discard(current)
                self._starting.discard(current)

        return main

    def _suspend(self, current: greenlet.greenlet) -> None:
        hub = self._hub_greenlet()
        if current in self._starting:
            self._starting.discard(current)
            spawner = current.parent
            current.parent = hub
            if spawner is None:
                raise RuntimeError("a started greenlet lost the greenlet that started it")
            spawner.switch()
        else:
            hub.switch()

    def _hub_greenlet(self) -> greenlet.greenlet:
        if self._hub is None:
            self._hub = greenlet.greenlet(self._run_hub, parent=self._owner)
        return self._hub

    def _run_hub(self) -> None:
        while not self._closing:
            self._ready.get().switch()
