"""`jit` and `map`.

`jit` changes when model calls wait, never what a function computes. Inside a
jitted function each model call is sent at its call site and returns a
pending result; reading the result waits for that call only. The function runs
once, in program order, on the caller's thread.

`map` runs a function on each item in its own greenlet, so the items' model
calls overlap. Items interleave only where one of them waits for a result.

Both end at a boundary: everything started inside has finished before they
return, and a failure that nobody read is raised there.
"""

import functools
import inspect
import threading
import types
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from typing import Generic, ParamSpec, TypeVar, overload

from ember.xcs import strands
from ember.xcs.frames import Call, Frame, Plan
from ember.xcs.graph import Graph

__all__ = ["Jitted", "jit", "map"]

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

_MAX_NOTED_FAILURES = 5


class Jitted(Generic[P, R]):
    """A function whose independent model calls overlap.

    Calling it runs the function once. `graph` returns what the most recent
    call executed. Nothing else carries over between calls: no result, no
    argument and no object the function created is kept.

    Raises:
        TypeError: On construction, for async functions and generator
            functions, whose bodies run after the call returns.
    """

    def __init__(self, fn: Callable[P, R]) -> None:
        if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
            raise TypeError(f"jit cannot run async function {fn.__qualname__}")
        if inspect.isgeneratorfunction(fn):
            raise TypeError(f"jit cannot run generator function {fn.__qualname__}")
        self._fn = fn
        self._lock = threading.Lock()
        self._graph: Graph | None = None
        self._plan: Plan | None = None
        functools.update_wrapper(self, fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            plan = self._plan
        with strands.scope(plan) as strand:
            frame = strand.frame
            try:
                result = _run_to_boundary(strand, functools.partial(self._fn, *args, **kwargs))
            except Exception:
                self._remember(frame)
                raise
            self._remember(frame)
            return result

    def graph(self) -> Graph | None:
        """What the most recent call executed, or ``None`` before the first call."""
        with self._lock:
            return self._graph

    @overload
    def __get__(self, instance: None, owner: type) -> "Jitted[P, R]": ...

    @overload
    def __get__(self, instance: object, owner: type) -> Callable[..., R]: ...

    def __get__(self, instance: object | None, owner: type) -> "Jitted[P, R] | Callable[..., R]":
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def _remember(self, frame: Frame) -> None:
        graph = frame.graph()
        plan = frame.next_plan(graph) if frame.parent is None else None
        with self._lock:
            self._graph = graph
            if plan is not None:
                self._plan = plan


def jit(fn: Callable[P, R]) -> Jitted[P, R]:
    """Makes ``fn``'s independent model calls overlap; see `Jitted`."""
    return Jitted(fn)


def map(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """Runs ``fn`` on each item concurrently and returns the results in order.

    Each item runs in its own greenlet on the caller's thread, with a copy of
    the caller's context. Items start in order, each running until it first
    waits before the next starts, so every item starts before any continues
    past a wait. If one raises, no further items start, the items already
    started finish, and the error of the earliest failed item is raised.
    """
    work = list(items)
    if not work:
        return []
    strand = strands.current()
    if strand is not None:
        return _map_in(strand, fn, work)
    with strands.scope(None) as strand:
        return _run_to_boundary(strand, lambda: _map_in(strand, fn, work))


def _run_to_boundary(strand: strands.Strand, body: Callable[[], R]) -> R:
    """Runs ``body``, then waits for everything started in the strand's frame.

    Raises the body's error if it raised; otherwise the first failure in the
    frame that nobody read. Either way the other unread failures are attached
    as notes.
    """
    frame = strand.frame
    try:
        result = body()
    except Exception as error:
        _note_failures(error, _settle(strand, frame))
        raise
    failures = _settle(strand, frame)
    if failures:
        first, *others = failures
        _note_failures(first, others)
        raise first
    return result


def _settle(strand: strands.Strand, frame: Frame) -> list[BaseException]:
    position = 0
    while position < len(frame.pending):
        strand.wait(frame.pending[position])
        position += 1
    return frame.take_unread_failures()


def _note_failures(error: BaseException, others: list[BaseException]) -> None:
    if not others:
        return
    shown = "; ".join(f"{type(other).__name__}: {other}" for other in others[:_MAX_NOTED_FAILURES])
    hidden = len(others) - _MAX_NOTED_FAILURES
    more = f" and {hidden} more" if hidden > 0 else ""
    error.add_note(f"{len(others)} other results also failed and were never read: {shown}{more}")


class _MapRun:
    """Shared state of one `map` call's items."""

    def __init__(self) -> None:
        self.failed = False
        self.reads: set[Call] = set()


def _map_in(strand: strands.Strand, fn: Callable[[T], R], items: list[T]) -> list[R]:
    run = _MapRun()
    outcomes: list[Future[R]] = []
    for item in items:
        if run.failed:
            break
        outcome: Future[R] = Future()
        outcomes.append(outcome)
        strand.spawn(functools.partial(_run_item, run, fn, item, outcome))
    for outcome in outcomes:
        strand.wait(outcome)
    strand.absorb(frozenset(run.reads))
    errors = [error for outcome in outcomes if (error := outcome.exception()) is not None]
    if errors:
        first, *others = errors
        _note_failures(first, others)
        raise first
    return [outcome.result() for outcome in outcomes]


def _run_item(
    run: _MapRun, fn: Callable[[T], R], item: T, outcome: "Future[R]", child: strands.Strand
) -> None:
    try:
        result = fn(item)
    except Exception as error:  # Raised by `map` in item order.
        run.failed = True
        outcome.set_exception(error)
    else:
        outcome.set_result(result)
    finally:
        run.reads |= child.reads
