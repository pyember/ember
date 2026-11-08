"""Observing every request attempt.

Code that needs a record of each request, such as an evaluation run, installs
an `Observer` for a block with `observing()`. The client reports every
attempt, including retries, to every observer active at the call site.
Observers run on whichever thread finished the attempt and should return
quickly.

An observer that raises cannot cost the caller a result it paid for, or hide
the attempt from other observers: every observer still receives the attempt,
the call's outcome is delivered as usual, and the observer's exceptions are
raised together, as an `ExceptionGroup`, when its `observing()` block exits.
An attempt reported after that block has exited raises the observer's
exception at once.

Every attempt of a call is reported before the call's outcome can be read:
before its `Completion` holds a result or an error, and before a stream yields
`Done` or raises. A caller that has read a call's outcome has therefore seen
all of that call's attempts.

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
import threading
from collections.abc import Generator
from typing import Protocol

from ember.errors import ModelError
from ember.llm.ref import ModelRef
from ember.llm.result import Result

__all__ = ["Attempt", "Observer", "Reporter", "active_observers", "observing", "reporter"]


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


class _Sink:
    """One `observing()` block: its observer and the exceptions it raised."""

    def __init__(self, observer: Observer) -> None:
        self.observer = observer
        self._lock = threading.Lock()
        self._errors: list[Exception] = []
        self._open = True

    def deliver(self, attempt: Attempt) -> None:
        try:
            self.observer.on_attempt(attempt)
        except Exception as error:  # Held for the block's exit; see the module docstring.
            with self._lock:
                if self._open:
                    self._errors.append(error)
                    return
            raise

    def close(self) -> list[Exception]:
        with self._lock:
            self._open = False
            errors, self._errors = self._errors, []
        return errors


@dataclasses.dataclass(frozen=True, slots=True)
class Reporter:
    """Delivers attempts to the observers that were active where it was made."""

    sinks: tuple[_Sink, ...]

    def report(self, attempt: Attempt) -> None:
        """Gives ``attempt`` to every observer, outermost first."""
        for sink in self.sinks:
            sink.deliver(attempt)


_SINKS: contextvars.ContextVar[tuple[_Sink, ...]] = contextvars.ContextVar(
    "ember_observers", default=()
)


def active_observers() -> tuple[Observer, ...]:
    """Returns the observers active in the current context, outermost first."""
    return tuple(sink.observer for sink in _SINKS.get())


def reporter() -> Reporter:
    """Returns a reporter for the observers active here, to use from any thread."""
    return Reporter(_SINKS.get())


@contextlib.contextmanager
def observing(observer: Observer) -> Generator[Observer, None, None]:
    """Reports every attempt made inside the block to ``observer``.

    Raises:
        ExceptionGroup: ``observer`` raised while receiving attempts; every
            exception it raised is in the group.
    """
    sink = _Sink(observer)
    token = _SINKS.set((*_SINKS.get(), sink))
    try:
        yield observer
    finally:
        _SINKS.reset(token)
        errors = sink.close()
        if errors:
            raise ExceptionGroup(f"observer {observer!r} raised while observing", errors)
