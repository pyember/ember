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

Tasks given to `dispatch` and `dispatch_stream` are leaf network I/O: they
never call back into ember and never wait on other ember work, so worker
threads cannot deadlock on each other.

Until the scheduler lands, everything runs synchronously on the caller's
thread, which is the behaviour outside `@xcs.jit` and `xcs.map` anyway.
"""

import collections
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from typing import Generic, TypeVar

__all__ = ["WaitQueue", "deferred", "dispatch", "dispatch_stream", "wait"]

T = TypeVar("T")


def dispatch(task: Callable[[], T], *, label: str) -> "Future[T]":
    """Starts ``task`` and returns a future for its result.

    Args:
        task: One request's network I/O. It must not call into ember.
        label: Short description for recorded graphs, such as the model id.
    """
    del label  # Recorded by the scheduler; unused when calls run synchronously.
    future: Future[T] = Future()
    try:
        future.set_result(task())
    except Exception as exc:  # Not handled here: `wait` re-raises it unchanged.
        future.set_exception(exc)
    return future


def dispatch_stream(source: Callable[[], Iterator[T]], *, label: str) -> Iterator[T]:
    """Starts a streaming request and returns an iterator over its items.

    The scheduler drains ``source`` on a worker thread; the caller reads items
    in order, and waiting for the next item lets other greenlets run. An error
    raised by ``source`` is raised by the iterator, unchanged, after the items
    that preceded it.

    Args:
        source: Opens the stream and yields its items. It must not call into
            ember.
        label: Short description for recorded graphs.
    """
    del label
    return source()


def deferred() -> bool:
    """Whether model calls made here return before they finish.

    True inside `@xcs.jit` and `xcs.map`: a call returns a pending result at
    once and a failure raises when that result is first read. False
    everywhere else: a call waits and raises at the call site, like any
    synchronous function. The answer depends only on scope, never on timing.
    """
    return False


def wait(future: "Future[T]") -> T:
    """Returns the result of ``future``, raising its exception unchanged."""
    return future.result()


class WaitQueue(Generic[T]):
    """Acquirers waiting for a shared capacity, such as a provider's permits.

    The scheduler decides who gets each freed slot. Until it lands, slots go
    in arrival order. Not thread-safe: the owner calls it under its own lock.
    """

    def __init__(self) -> None:
        self._items: collections.deque[T] = collections.deque()

    def push(self, item: T) -> None:
        """Adds ``item``, for the request the caller sends next."""
        self._items.append(item)

    def pop(self) -> T:
        """Removes and returns the item that gets the freed slot.

        Raises:
            IndexError: The queue is empty.
        """
        if not self._items:
            raise IndexError("pop from an empty WaitQueue")
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)
