"""The boundary between model calls and the xcs scheduler.

`ember.llm` uses exactly four functions from this module:

- `wait(future)` to read any result, including a rate-limit permit. Inside
  `@xcs.jit` and `xcs.map` it lets other greenlets run while this one waits.
- `dispatch(task)` to run one request that already holds its permit.
- `dispatch_stream(source)` to run one streaming request that already holds
  its permit, reading its items as they arrive.
- `deferred()` to decide where a failed call raises: at the call site when
  it returns False, at the first read of its result when it returns True.

The limiter keeps the acquirers waiting for capacity in a `WaitQueue`, which
decides who gets each freed slot.

Inside `@xcs.jit` and `xcs.map`, `dispatch` returns a pending future at once
and waiting lets other greenlets run. Outside them everything runs
synchronously on the caller's thread, like plain Python.

Tasks given to `dispatch` and `dispatch_stream` are leaf network I/O: they
never call back into ember and never wait on other ember work, so worker
threads cannot deadlock on each other. Each runs in a copy of the context
that was active where it was dispatched.
"""

import contextvars
import heapq
import itertools
import time
import weakref
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future
from typing import Generic, TypeVar

from ember.xcs import strands, workers
from ember.xcs.frames import Call, Pending
from ember.xcs.streams import Channel, Ended, Waiting

__all__ = ["WaitQueue", "deferred", "dispatch", "dispatch_stream", "wait"]

T = TypeVar("T")


def dispatch(task: Callable[[], T], *, label: str) -> "Future[T]":
    """Starts ``task`` and returns a future for its result.

    Inside a scope the task runs on a worker thread and the future is
    pending; outside one the task has already run when this returns.

    Args:
        task: One request's network I/O. It must not call into ember.
        label: Short description for recorded graphs, such as the model id.

    Raises:
        RuntimeError: Inside a scope, no worker thread could start for the
            task; it was not sent and nothing waits for it.
    """
    strand = strands.current()
    if strand is None:
        return _run_now(task)
    call = strand.next_call(label)
    pending: Pending[T] = Pending(call)
    context = contextvars.copy_context()
    workers.submit(lambda: _run_call(call, pending, context, task))
    strand.record(call, pending)
    return pending


def dispatch_stream(source: Callable[[], Iterator[T]], *, label: str) -> Iterator[T]:
    """Starts a streaming request and returns an iterator over its items.

    Inside a scope a worker drains ``source`` as items arrive, and waiting for
    the next item lets other greenlets run. An error raised by ``source`` is
    raised by the iterator, unchanged, after the items that preceded it.
    Closing the iterator, or dropping it, whether or not reading has started,
    abandons the stream: the worker stops when the next item arrives (until
    then the enclosing jit boundary waits for it and the request keeps its
    permit), and a failure the reader did not receive is not raised.
    Outside a scope this returns ``source()`` itself.

    Args:
        source: Opens the stream and yields its items. It must not call into
            ember.
        label: Short description for recorded graphs.

    Raises:
        RuntimeError: Inside a scope, no worker thread could start for the
            stream; it was not opened and nothing waits for it.
    """
    strand = strands.current()
    if strand is None:
        return source()
    call = strand.next_call(label)
    drained: Pending[None] = Pending(call)
    channel: Channel[T] = Channel()
    context = contextvars.copy_context()
    workers.submit(lambda: _drain(call, drained, channel, context, source))
    strand.record(call, drained)
    return _Reader(channel, drained)


def deferred() -> bool:
    """Whether model calls made here return before they finish.

    True inside `@xcs.jit` and `xcs.map`: a call returns a pending result at
    once and a failure raises when that result is first read. False
    everywhere else, including on worker threads: a call waits and raises at
    the call site, like any synchronous function. The answer depends only on
    scope, never on timing.
    """
    return strands.current() is not None


def wait(future: "Future[T]") -> T:
    """Returns the result of ``future``, raising its exception unchanged.

    Inside a scope other greenlets run until ``future`` is done.
    """
    strand = strands.current()
    if strand is not None:
        strand.wait(future)
    if isinstance(future, Pending):
        future.read = True
        if strand is not None:
            strand.absorb(frozenset({future.call}))
    return future.result()


class WaitQueue(Generic[T]):
    """Acquirers waiting for a shared capacity, such as a provider's permits.

    Slots go to jit calls in the order their acquirers arrived, as in a FIFO
    queue, so no call can starve another. Within one jit call, each slot goes
    to the waiting acquirer whose next request has the longest predicted
    remaining path, from the function's previous run (see `Jitted.graph`);
    without a prediction, in arrival order. Outside jit and map, every
    acquirer is its own call, so the queue is plain FIFO.

    Not thread-safe: the owner calls it under its own lock.
    """

    def __init__(self) -> None:
        self._serial = itertools.count()
        self._arrivals: list[tuple[int, object]] = []
        self._by_scope: dict[object, list[tuple[float, int, T]]] = {}

    def push(self, item: T) -> None:
        """Adds ``item``, for the request the running greenlet sends next.

        Call it on the acquiring greenlet: that is how the queue knows which
        jit call and which request is waiting.
        """
        serial = next(self._serial)
        strand = strands.current()
        if strand is None:
            scope: object = object()
            priority = 0.0
        else:
            scope = strand.frame.outermost()
            priority = strand.next_call_remaining_s()
        heapq.heappush(self._arrivals, (serial, scope))
        heapq.heappush(self._by_scope.setdefault(scope, []), (-priority, serial, item))

    def pop(self) -> T:
        """Removes and returns the item that gets the freed slot.

        Any thread may call it, such as the worker whose task freed the slot.

        Raises:
            IndexError: The queue is empty.
        """
        if not self._arrivals:
            raise IndexError("pop from an empty WaitQueue")
        _, scope = heapq.heappop(self._arrivals)
        waiting = self._by_scope[scope]
        _, _, item = heapq.heappop(waiting)
        if not waiting:
            del self._by_scope[scope]
        return item

    def __len__(self) -> int:
        return len(self._arrivals)


def _run_now(task: Callable[[], T]) -> "Future[T]":
    future: Future[T] = Future()
    try:
        future.set_result(task())
    except Exception as error:  # Re-raised unchanged by `wait`.
        future.set_exception(error)
    return future


def _run_call(
    call: Call, pending: "Pending[T]", context: contextvars.Context, task: Callable[[], T]
) -> None:
    try:
        result = context.run(task)
    # A worker must never die with an unreported error, so whatever the task raises
    # (SystemExit included) is relayed to the reader, as concurrent.futures does.
    except BaseException as error:
        call.finished_at = time.monotonic()
        pending.set_exception(error)
    else:
        call.finished_at = time.monotonic()
        pending.set_result(result)


def _drain(
    call: Call,
    drained: "Pending[None]",
    channel: Channel[T],
    context: contextvars.Context,
    source: Callable[[], Iterator[T]],
) -> None:
    error: BaseException | None = None
    try:
        context.run(_pump, channel, source)
    # Relayed to the reader after the items before it, for the same reason as `_run_call`.
    except BaseException as caught:
        error = caught
    call.finished_at = time.monotonic()
    channel.end(error)
    if error is None:
        drained.set_result(None)
    else:
        drained.set_exception(error)


def _pump(channel: Channel[T], source: Callable[[], Iterator[T]]) -> None:
    items = source()
    try:
        for item in items:
            if not channel.put(item):
                return
    finally:
        if isinstance(items, Generator):
            items.close()


class _Reader(Iterator[T]):
    """Reads one stream's items. Closing or dropping it, before or after
    reading starts, abandons the stream."""

    def __init__(self, channel: Channel[T], drained: "Pending[None]") -> None:
        self._channel = channel
        self._drained = drained
        self._open = True
        self._abandon = weakref.finalize(self, _abandon, channel, drained)

    def __next__(self) -> T:
        while self._open:
            step = self._channel.take()
            if isinstance(step, Waiting):
                wait(step.future)
            elif isinstance(step, Ended):
                self.close()
                if step.error is not None:
                    raise step.error
            else:
                strand = strands.current()
                if strand is not None:
                    strand.absorb(frozenset({self._drained.call}))
                return step.value
        raise StopIteration

    def close(self) -> None:
        """Abandons the stream; the worker stops when the next item arrives."""
        self._open = False
        self._abandon()


def _abandon(channel: Channel[T], drained: "Pending[None]") -> None:
    drained.read = True
    channel.close()
