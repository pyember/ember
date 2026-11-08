"""Each greenlet's place in a jit or map scope.

A `Strand` is the scheduler's record of one greenlet: the frame it runs in,
the calls whose results it has read, and its position in the program. The
running greenlet finds its strand through a contextvar. Every greenlet gets
its own copy of its spawner's context, so the lookup also checks that the
strand belongs to the running greenlet: code on another thread or in a
greenlet ember did not start is outside the scope, even if it copied the
context.
"""

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from typing import TypeVar

import greenlet

from ember.xcs.frames import Call, Frame, Plan
from ember.xcs.scheduler import Scheduler

__all__ = ["Strand", "current", "scope"]

T = TypeVar("T")

_CURRENT: contextvars.ContextVar["Strand | None"] = contextvars.ContextVar(
    "ember_xcs_strand", default=None
)


class Strand:
    """One greenlet inside a scope.

    Attributes:
        scheduler: The scope's scheduler.
        greenlet: The greenlet this strand describes.
        frame: The innermost frame the greenlet runs in.
        path: Spawn positions from the scope's first strand to this one.
        reads: Calls whose results the greenlet has read, directly or through
            other results; a call it dispatches depends on these.
        dispatched: How many calls the greenlet has dispatched.
        spawned: How many strands the greenlet has started.
    """

    __slots__ = ("dispatched", "frame", "greenlet", "path", "reads", "scheduler", "spawned")

    def __init__(
        self,
        scheduler: Scheduler,
        frame: Frame,
        path: tuple[int, ...],
        reads: frozenset[Call],
    ) -> None:
        self.scheduler = scheduler
        self.greenlet = greenlet.getcurrent()
        self.frame = frame
        self.path = path
        self.reads = reads
        self.dispatched = 0
        self.spawned = 0

    def wait(self, future: "Future[T]") -> None:
        """Lets other greenlets run until ``future`` is done."""
        self.scheduler.park(future)

    def absorb(self, calls: frozenset[Call]) -> None:
        """Records that the greenlet has read results depending on ``calls``.

        A call implied by another one read here (because that one depended
        on it) is dropped, which keeps a long sequential chain at one read.
        """
        implied = {dep for call in calls for dep in call.deps}
        self.reads = (self.reads - implied) | calls

    def next_call(self, label: str) -> Call:
        """Records a call this greenlet is dispatching now."""
        call = Call(label, self.reads, (self.path, self.dispatched))
        self.dispatched += 1
        self.frame.record(call)
        return call

    def next_call_remaining_s(self) -> float:
        """The current plan's prediction for the next call this greenlet dispatches."""
        return self.frame.planned_remaining_s((self.path, self.dispatched))

    def spawn(self, body: Callable[["Strand"], None]) -> None:
        """Runs ``body`` in a new greenlet, in a copy of this greenlet's context.

        The new strand starts with this strand's frame and reads. ``body``
        runs until it first waits, then this greenlet continues.
        """
        path = (*self.path, self.spawned)
        self.spawned += 1
        frame, reads = self.frame, self.reads

        def run() -> None:
            child = Strand(self.scheduler, frame, path, reads)
            _CURRENT.set(child)
            body(child)

        self.scheduler.start(run, contextvars.copy_context())


def current() -> Strand | None:
    """The running greenlet's strand, or ``None`` outside any scope."""
    strand = _CURRENT.get()
    if strand is None or strand.greenlet is not greenlet.getcurrent():
        return None
    return strand


@contextlib.contextmanager
def scope(plan: Plan | None) -> Iterator[Strand]:
    """Runs a block in a new frame, opening a scope if none is active.

    Inside an active scope the new frame nests in the running strand's frame
    and ``plan`` is ignored: only the outermost frame plans. Otherwise a new
    scheduler starts, the block runs on the calling greenlet as the scope's
    first strand, and the scheduler closes when the block ends.

    Args:
        plan: The previous run's plan, used if this frame opens the scope.

    Yields:
        The running strand, whose ``frame`` is the new frame.
    """
    strand = current()
    if strand is not None:
        enclosing = strand.frame
        strand.frame = Frame(parent=enclosing, plan=None)
        try:
            yield strand
        finally:
            strand.frame = enclosing
        return
    scheduler = Scheduler()
    first = Strand(scheduler, Frame(parent=None, plan=plan), path=(), reads=frozenset())
    token = _CURRENT.set(first)
    try:
        yield first
    finally:
        _CURRENT.reset(token)
        scheduler.close()
