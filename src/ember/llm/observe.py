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
Calls deferred inside ``@xcs.jit`` can finish after that block exits; an
observer that raises then is reported through `sys.unraisablehook`, the channel
for errors that have nowhere to propagate, naming the observer and the call.

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
import sys
import threading
import traceback
import types
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
        except Exception as error:  # Never raised into the call; see the module docstring.
            with self._lock:
                if self._open:
                    self._errors.append(error)
                    return
            _report_unraisable(
                error,
                f"Exception ignored in observer {self.observer!r} for call {attempt.call_id}, "
                "after its observing() block exited",
                self.observer,
            )

    def close(self) -> list[Exception]:
        with self._lock:
            self._open = False
            errors, self._errors = self._errors, []
        return errors


@dataclasses.dataclass(slots=True)
class _Unraisable:
    """The fields `sys.unraisablehook` reads from its argument."""

    exc_type: type[BaseException]
    exc_value: BaseException | None
    exc_traceback: types.TracebackType | None
    err_msg: str | None
    object: object


def _report_unraisable(error: Exception, message: str, source: object) -> None:
    hook = sys.unraisablehook
    if hook is sys.__unraisablehook__:
        # The built-in hook accepts only an argument type Python does not
        # expose, so print what it would print.
        print(f"{message}: {source!r}", file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        return
    hook(_Unraisable(type(error), error, error.__traceback__, message, source))


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
