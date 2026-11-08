"""Shared fixtures. No test reaches the network."""

import contextlib
import dataclasses
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ember import context
from ember.context import RetryPolicy
from ember.errors import ErrorKind
from ember.llm import Client, using
from ember.llm.fake import Answer, Fail, FakeProvider, Reply
from ember.llm.ref import ModelRef
from ember.llm.request import Message, Request, TextPart

ReplyFn = Callable[[str, "str | tuple[Message, ...]"], "str | Reply | ErrorKind"]


@dataclasses.dataclass(frozen=True)
class SentCall:
    """One request the fake received, with the fields operators set."""

    model: str
    prompt: "str | tuple[Message, ...]"
    system: str | None
    temperature: float | None
    max_output_tokens: int | None
    seed: int | None


class FakeModels:
    """A `FakeProvider` that answers with ``reply(model, prompt)``.

    ``reply`` returns the answer text, a whole `Reply` (to set its finish
    reason, for example), or an `ErrorKind` for the call to fail with. Each
    answer takes ``latency_s``, slept here on the thread that runs the request
    so that `peak_in_flight` counts requests answered at once.
    """

    def __init__(self, reply: ReplyFn, latency_s: float) -> None:
        self._reply = reply
        self._latency_s = latency_s
        self._lock = threading.Lock()
        self._in_flight = 0
        self._peak_in_flight = 0
        self.provider = FakeProvider(respond=self._respond)

    @property
    def calls(self) -> list[SentCall]:
        """Requests received so far, in arrival order."""
        return [_sent(seen.model, seen.request) for seen in self.provider.requests]

    @property
    def peak_in_flight(self) -> int:
        """The most requests that were being answered at the same moment."""
        with self._lock:
            return self._peak_in_flight

    def _respond(self, model: ModelRef, request: Request) -> Answer:
        with self._lock:
            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
        try:
            time.sleep(self._latency_s)
            outcome = self._reply(str(model), _prompt_of(request))
        finally:
            with self._lock:
                self._in_flight -= 1
        if isinstance(outcome, ErrorKind):
            return Fail(outcome)
        return outcome if isinstance(outcome, Reply) else Reply(outcome)


def _sent(model: ModelRef, request: Request) -> SentCall:
    sampling = request.sampling
    return SentCall(
        str(model),
        _prompt_of(request),
        request.system,
        sampling.temperature,
        sampling.max_output_tokens,
        sampling.seed,
    )


def _prompt_of(request: Request) -> "str | tuple[Message, ...]":
    """A single-turn text request's prompt is its text; any other's, its messages."""
    [*_, last] = request.messages
    if len(request.messages) == 1 and len(last.parts) == 1 and isinstance(last.parts[0], TextPart):
        return last.parts[0].text
    return request.messages


@pytest.fixture
def fake_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., FakeModels]]:
    """Installs a client whose every provider answers with a reply function.

    Config is isolated to a temporary directory, and the client makes one
    attempt per call so that each test sees exactly the calls operators make.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EMBER_CONFIG", str(tmp_path / "config.yaml"))
    context.reload()
    with contextlib.ExitStack() as stack:

        def install(reply: ReplyFn, latency_s: float = 0.0) -> FakeModels:
            fake = FakeModels(reply, latency_s)
            client = stack.enter_context(Client(provider_factory=fake.provider.factory))
            stack.enter_context(using(client))
            stack.enter_context(context.override(retry=RetryPolicy(max_attempts=1)))
            return fake

        yield install
    context.reload()
