"""The boundary between model calls and the xcs scheduler.

`ember.llm` uses exactly three functions from this module:

- `wait(future)` to read any result, including a rate-limit permit. Inside
  `@xcs.jit` and `xcs.map` it lets other greenlets run while this one waits.
- `dispatch(task)` to run one request that already holds its permit.
- `dispatch_stream(source)` to run one streaming request that already holds
  its permit, reading its items as they arrive.

Tasks given to `dispatch` and `dispatch_stream` are leaf network I/O: they
never call back into ember and never wait on other ember work, so worker
threads cannot deadlock on each other.

Until the scheduler lands, everything runs synchronously on the caller's
thread, which is the behaviour outside `@xcs.jit` and `xcs.map` anyway.
"""

from collections.abc import Callable, Iterator
from concurrent.futures import Future
from typing import TypeVar

__all__ = ["dispatch", "dispatch_stream", "wait"]

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


def wait(future: "Future[T]") -> T:
    """Returns the result of ``future``, raising its exception unchanged."""
    return future.result()
