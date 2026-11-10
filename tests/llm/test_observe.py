"""Observers: an observer's error never reaches the call that reported to it."""

import sys

import pytest

from ember.errors import ErrorKind, ModelError
from ember.llm import observe
from ember.llm.observe import Attempt, observing
from ember.llm.ref import ModelRef

ATTEMPT = Attempt(
    call_id="call-7",
    number=1,
    model=ModelRef("openai", "gpt-5.5"),
    latency_s=0.0,
    outcome=ModelError(ErrorKind.SERVER, provider="openai", model="gpt-5.5", reason="busy"),
    will_retry=False,
)


class Broken:
    def __init__(self) -> None:
        self.error = ValueError("observer bug")

    def on_attempt(self, attempt: Attempt) -> None:
        raise self.error

    def __repr__(self) -> str:
        return "Broken()"


class Recorder:
    def __init__(self) -> None:
        self.attempts: list[Attempt] = []

    def on_attempt(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)


def test_an_error_after_the_block_exits_goes_to_the_unraisable_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", received.append)
    broken, recorder = Broken(), Recorder()
    with observing(broken), observing(recorder):
        report = observe.reporter()
    report.report(ATTEMPT)
    assert recorder.attempts == [ATTEMPT]
    [unraisable] = received
    assert unraisable.exc_value is broken.error  # type: ignore[attr-defined]
    assert unraisable.object is broken  # type: ignore[attr-defined]
    message = unraisable.err_msg  # type: ignore[attr-defined]
    assert "Broken()" in message and "call-7" in message


def test_with_the_built_in_hook_the_error_is_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "unraisablehook", sys.__unraisablehook__)
    with observing(Broken()):
        report = observe.reporter()
    report.report(ATTEMPT)
    printed = capsys.readouterr().err
    assert "Exception ignored in observer Broken() for call call-7" in printed
    assert "ValueError: observer bug" in printed


def test_errors_inside_the_block_raise_together_when_it_exits() -> None:
    broken, recorder = Broken(), Recorder()
    with pytest.raises(ExceptionGroup) as caught:
        with observing(broken), observing(recorder):
            observe.reporter().report(ATTEMPT)
            observe.reporter().report(ATTEMPT)
    assert caught.value.exceptions == (broken.error, broken.error)
    assert recorder.attempts == [ATTEMPT, ATTEMPT]
