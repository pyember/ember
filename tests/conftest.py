"""Shared fixtures. No test reaches the network."""

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future

import pytest

from ember.errors import ModelError
from ember.llm.completion import Completion
from ember.llm.ref import ModelRef
from ember.llm.request import Message
from ember.llm.result import FinishReason, Result, Usage
from ember.operators import model_call

Reply = Callable[[str, "str | tuple[Message, ...]"], "str | ModelError"]


@dataclasses.dataclass(frozen=True)
class SentCall:
    """One call the fake received, with the arguments operators pass."""

    model: str
    prompt: "str | tuple[Message, ...]"
    system: str | None
    temperature: float | None
    max_output_tokens: int | None
    seed: int | None


@dataclasses.dataclass
class FakeModels:
    """Stands in for `ember.models`: records calls and answers with `reply`.

    `reply` returns the text to answer with, or a `ModelError` to fail with.
    """

    reply: Reply
    calls: list[SentCall] = dataclasses.field(default_factory=list)

    def __call__(
        self,
        model: str | ModelRef,
        prompt: "str | Sequence[Message]",
        *,
        system: str | None,
        temperature: float | None,
        top_p: float | None,
        max_output_tokens: int | None,
        stop: Sequence[str],
        seed: int | None,
        reasoning_effort: str | None,
    ) -> Completion:
        del top_p, stop, reasoning_effort
        sent = prompt if isinstance(prompt, str) else tuple(prompt)
        self.calls.append(SentCall(str(model), sent, system, temperature, max_output_tokens, seed))
        outcome = self.reply(str(model), sent)
        if isinstance(outcome, ModelError):
            raise outcome
        ref = model if isinstance(model, ModelRef) else ModelRef.parse(model)
        future: Future[Result] = Future()
        future.set_result(result_of(outcome, ref))
        return Completion(future, ref)


def result_of(text: str, model: ModelRef) -> Result:
    """A finished result carrying ``text``."""
    return Result(
        text=text,
        message=Message.assistant(text),
        reasoning="",
        tool_calls=(),
        finish_reason=FinishReason.STOP,
        usage=Usage(input=3, output=2),
        cost=None,
        model=model,
        request_id="req-fake",
        latency_s=0.0,
    )


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Reply], FakeModels]]:
    """Installs a `FakeModels` built from a reply function."""

    def install(reply: Reply) -> FakeModels:
        fake = FakeModels(reply)
        monkeypatch.setattr(model_call, "models", fake)
        return fake

    yield install
