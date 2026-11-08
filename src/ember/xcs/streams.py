"""Carrying one stream's items from the worker that drains it to its reader.

The worker never waits for the reader: it appends each item and moves on.
That keeps the rule that worker threads never wait on other ember work, which
a fixed-size buffer would break (a full buffer would hold a worker hostage to
a reader that may itself be waiting for a queued request). The buffer is
bounded by the response, the same memory a non-streaming result holds, and
the reader closing the channel stops the worker after its current item.
"""

import collections
import dataclasses
import threading
from concurrent.futures import Future
from typing import Generic, TypeAlias, TypeVar

__all__ = ["Channel", "Ended", "Item", "Step", "Waiting"]

T = TypeVar("T")


@dataclasses.dataclass(frozen=True, slots=True)
class Item(Generic[T]):
    """The next item."""

    value: T


@dataclasses.dataclass(frozen=True, slots=True)
class Waiting:
    """No item yet; ``future`` completes when there is one or the stream ends."""

    future: "Future[None]"


@dataclasses.dataclass(frozen=True, slots=True)
class Ended:
    """The stream is over; ``error`` is why, if it failed."""

    error: BaseException | None


Step: TypeAlias = Item[T] | Waiting | Ended


class Channel(Generic[T]):
    """One producer thread, one reader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: collections.deque[T] = collections.deque()
        self._ended: Ended | None = None
        self._closed = False
        self._waiter: Future[None] | None = None

    def put(self, item: T) -> bool:
        """Adds ``item``. Returns False once the reader has closed the channel."""
        with self._lock:
            if self._closed:
                return False
            self._items.append(item)
            waiter, self._waiter = self._waiter, None
        if waiter is not None:
            waiter.set_result(None)
        return True

    def end(self, error: BaseException | None) -> None:
        """Marks the stream finished, successfully or with ``error``."""
        with self._lock:
            self._ended = Ended(error)
            waiter, self._waiter = self._waiter, None
        if waiter is not None:
            waiter.set_result(None)

    def take(self) -> Step[T]:
        """The next item, the end of the stream, or a future to wait on first."""
        with self._lock:
            if self._items:
                return Item(self._items.popleft())
            if self._ended is not None:
                return self._ended
            if self._waiter is None:
                self._waiter = Future()
            return Waiting(self._waiter)

    def close(self) -> None:
        """Stops delivery: the producer stops after its current item."""
        with self._lock:
            self._closed = True
            self._items.clear()
