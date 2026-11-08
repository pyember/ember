"""Observing every request attempt.

Code that needs a record of each request, such as an evaluation run, installs
an `Observer` for a block with `observing()`. The client reports every
attempt, including retries, to every observer active at the call site.
Observers run on whichever thread finished the attempt and must return quickly
and never raise.

Example:
    >>> from ember.llm.observe import Attempt, observing
    >>> class Collect:
    ...     def __init__(self) -> None:
    ...         self.attempts: list[Attempt] = []
    ...     def on_attempt(self, attempt: Attempt) -> None:
    ...         self.attempts.append(attempt)
    >>> collector = Collect()
    >>> with observing(collector):
    ...     pass  # model calls made here are reported to collector
"""

import contextlib
import contextvars
import dataclasses
from collections.abc import Generator
from typing import Protocol

from ember.errors import ModelError
from ember.llm.ref import ModelRef
from ember.llm.result import Result

__all__ = ["Attempt", "Observer", "active_observers", "observing"]


@dataclasses.dataclass(frozen=True, slots=True)
class Attempt:
    """One attempt to send one request.

    Attributes:
        call_id: Identifies the model call; retries of one call share it.
        number: 1 for the first attempt, 2 for the first retry, and so on.
        model: The model the attempt was sent to.
        latency_s: Seconds from sending to the last byte or the failure.
        outcome: The result, or the error that ended the attempt.
        will_retry: Whether the client will send another attempt.
    """

    call_id: str
    number: int
    model: ModelRef
    latency_s: float
    outcome: Result | ModelError
    will_retry: bool


class Observer(Protocol):
    """Receives every request attempt made while it is active."""

    def on_attempt(self, attempt: Attempt) -> None:
        """Called once per attempt, after it finishes."""
        ...


_OBSERVERS: contextvars.ContextVar[tuple[Observer, ...]] = contextvars.ContextVar(
    "ember_observers", default=()
)


def active_observers() -> tuple[Observer, ...]:
    """Returns the observers active in the current context, outermost first."""
    return _OBSERVERS.get()


@contextlib.contextmanager
def observing(observer: Observer) -> Generator[Observer, None, None]:
    """Reports every attempt made inside the block to ``observer``."""
    token = _OBSERVERS.set((*_OBSERVERS.get(), observer))
    try:
        yield observer
    finally:
        _OBSERVERS.reset(token)
