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
time the running one parks. The hub lives as long as the scope: an interrupt
such as KeyboardInterrupt that arrives while the thread waits in the hub is
raised in the owner, which the hub only ever runs for while the owner is
parked, and the scope stays usable if the owner catches it.

Invariant: each suspension is resumed exactly once. A parked greenlet is woken
by its future's callback with a token for that suspension, and the greenlet
withdraws the token as soon as the suspension ends, however it ends (resumed,
interrupted, or stopped), so any later wake for it is dropped. An interrupt
can land between the hub taking a wake from the queue and switching to it, so
after an interrupt the hub queues again every parked greenlet whose future is
done; the token makes a duplicate harmless. A child started by `start` returns
straight to the greenlet that started it the first time it parks.

`hold` keeps a set of children, and every greenlet they started, from
resuming: their wakes wait until `stop` reaches each one. A map holds all its
items before it stops any, so while one item unwinds (and perhaps waits in a
``finally`` block), the others cannot run.
"""

import contextvars
import os
import queue
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from typing import Any, TypeVar

import greenlet

__all__ = ["Scheduler"]

T = TypeVar("T")


class Scheduler:
    """Runs greenlets on the thread that created it, one at a time.

    Create it on the greenlet that will own the scope, and call `close` on the
    same greenlet when the scope ends.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._owner = greenlet.getcurrent()
        self._ready: queue.SimpleQueue[tuple[greenlet.greenlet, object]] = queue.SimpleQueue()
        # Results of every type the scope's code waits on: user data.
        self._parked: dict[greenlet.greenlet, tuple[object, Future[Any]]] = {}
        self._hub: greenlet.greenlet | None = None
        self._children: set[greenlet.greenlet] = set()
        self._spawners: dict[greenlet.greenlet, greenlet.greenlet] = {}
        self._starting: set[greenlet.greenlet] = set()
        self._held: set[greenlet.greenlet] = set()
        self._deferred: list[tuple[greenlet.greenlet, object]] = []
        self._closing = False
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the scope has ended; nothing may run on this scheduler after it."""
        return self._closed

    def park(self, future: "Future[T]") -> None:
        """Suspends the running greenlet until ``future`` is done.

        Other greenlets run meanwhile. Returns at once if ``future`` is done.
        An exception thrown into the greenlet while it is parked (an
        interrupt, or `stop`) is raised here, and the future's later wake is
        dropped.

        Raises:
            RuntimeError: The process forked inside the scope. The parent's
                workers do not exist in the child, so nothing would wake it.
        """
        if future.done():
            return
        if os.getpid() != self._pid:
            raise RuntimeError("a jit or map scope cannot wait in a process forked inside it")
        current = greenlet.getcurrent()
        token = object()
        self._parked[current] = (token, future)
        future.add_done_callback(lambda _: self._ready.put((current, token)))
        try:
            self._suspend(current)
        finally:
            del self._parked[current]

    def start(self, run: Callable[[], None], context: contextvars.Context) -> greenlet.greenlet:
        """Runs ``run`` in a new greenlet until it first parks or finishes.

        Control then returns to the caller; the new greenlet resumes later,
        when what it parked on is done. ``run`` executes in ``context`` and
        must not raise: it reports its own outcome.

        Returns:
            The new greenlet, for `stop`.
        """
        spawner = greenlet.getcurrent()
        child = greenlet.greenlet(self._child_main(run), parent=spawner)
        child.gr_context = context
        self._children.add(child)
        self._spawners[child] = spawner
        self._starting.add(child)
        child.switch()
        return child

    def hold(self, children: Iterable[greenlet.greenlet]) -> None:
        """Keeps ``children``, and every greenlet they started, from resuming until stopped."""
        self._held.update(child for child in children if not child.dead)

    def stop(self, child: greenlet.greenlet) -> None:
        """Ends a suspended child started by `start` and returns once it has finished.

        `greenlet.GreenletExit` is raised in the child where it is parked, so
        its ``finally`` blocks run. If the child parks again while unwinding,
        this waits for it like any other greenlet. Releases the child from
        `hold`. Does nothing if the child has finished.
        """
        self._held.discard(child)
        if child.dead:
            return
        self._release_deferred()
        child.parent = greenlet.getcurrent()
        child.throw(greenlet.GreenletExit)

    def close(self) -> None:
        """Ends the scope, on the owner, after its body has finished.

        Greenlets still suspended (possible only after an interrupt) are ended
        with `stop`.
        """
        self._closed = True
        for child in list(self._children):
            self.stop(child)
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
                self._spawners.pop(current, None)
                self._held.discard(current)
                self._starting.discard(current)

        return main

    def _suspend(self, current: greenlet.greenlet) -> None:
        hub = self._hub_greenlet()
        if hub.dead:
            raise RuntimeError("the scope's scheduler has stopped")
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
            try:
                self._resume_next()
            except greenlet.GreenletExit:
                raise
            # Not absorbed: an interrupt that arrives while the thread waits here
            # belongs to the owner, which is parked; the hub carries on afterwards.
            except BaseException as interrupt:
                self._queue_done_waits()
                self._owner.throw(interrupt)

    def _resume_next(self) -> None:
        parked, token = self._ready.get()
        waiting = self._parked.get(parked)
        if waiting is None or waiting[0] is not token:
            return
        if self._is_held(parked):
            self._deferred.append((parked, token))
            return
        parked.switch()

    def _is_held(self, candidate: greenlet.greenlet) -> bool:
        current: greenlet.greenlet | None = candidate
        while current is not None:
            if current in self._held:
                return True
            current = self._spawners.get(current)
        return False

    def _release_deferred(self) -> None:
        deferred, self._deferred = self._deferred, []
        for wake in deferred:
            self._ready.put(wake)

    def _queue_done_waits(self) -> None:
        for parked, (token, future) in list(self._parked.items()):
            if future.done():
                self._ready.put((parked, token))
