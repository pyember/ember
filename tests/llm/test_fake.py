"""FakeProvider: answers chosen per call, recorded requests, real latency."""

import pytest

import ember
from ember.errors import ErrorKind, ModelError
from ember.llm import Client, ModelRef, using
from ember.llm.fake import Fail, FakeProvider, Reply
from ember.llm.request import Request, TextPart
from ember.llm.result import Usage


def _prompt(request: Request) -> str:
    part = request.messages[-1].parts[0]
    assert isinstance(part, TextPart)
    return part.text


def test_respond_answers_by_model_and_prompt_through_ember_models() -> None:
    def respond(model: ModelRef, request: Request) -> Reply | Fail:
        if model.provider == "anthropic":
            return Fail(ErrorKind.PERMISSION, status=403)
        return Reply(f"{model.name}: {_prompt(request).upper()}", usage=Usage(input=2, output=1))

    fake = FakeProvider(respond=respond)
    with Client(provider_factory=fake.factory) as client, using(client):
        answer = ember.models("gpt-5.5", "abc", system="Shout.")
        with pytest.raises(ModelError) as caught:
            ember.models("claude-opus-5-5", "abc")
    assert (answer.text, answer.usage) == ("gpt-5.5: ABC", Usage(input=2, output=1))
    assert (caught.value.kind, caught.value.status, caught.value.provider) == (
        ErrorKind.PERMISSION,
        403,
        "anthropic",
    )
    seen = fake.requests
    assert [(s.model, s.streamed) for s in seen] == [
        (ModelRef("openai", "gpt-5.5"), False),
        (ModelRef("anthropic", "claude-opus-5-5"), False),
    ]
    assert (seen[0].request.system, _prompt(seen[0].request)) == ("Shout.", "abc")


def test_a_failure_carries_its_retry_after_to_the_client() -> None:
    sleeps: list[float] = []
    fake = FakeProvider([Fail(ErrorKind.RATE_LIMITED, status=429, retry_after_s=3.0), Reply("ok")])
    with Client(provider_factory=fake.factory, sleep=sleeps.append, jitter=lambda: 0.0) as client:
        assert client.complete("gpt-5.5", "hi").text == "ok"
    assert sleeps == [3.0]


def test_latency_is_spent_inside_the_request() -> None:
    fake = FakeProvider(default=Reply("slow", latency_s=0.05))
    with Client(provider_factory=fake.factory) as client:
        assert client.complete("gpt-5.5", "hi").latency_s >= 0.05


def test_script_answers_in_order_then_falls_back_to_default() -> None:
    fake = FakeProvider([Reply("one"), Reply("two")], default=Reply("rest"))
    with Client(provider_factory=fake.factory) as client:
        texts = [client.complete("gpt-5.5", "hi").text for _ in range(4)]
    assert texts == ["one", "two", "rest", "rest"]


def test_a_used_up_script_without_default_fails_loudly() -> None:
    fake = FakeProvider([Reply("only")], default=None)
    with Client(provider_factory=fake.factory) as client:
        client.complete("gpt-5.5", "hi")
        with pytest.raises(AssertionError, match="used up"):
            client.complete("gpt-5.5", "hi")


def test_script_and_respond_cannot_be_combined() -> None:
    with pytest.raises(ValueError, match="not both"):
        FakeProvider([Reply()], respond=lambda model, request: Reply())


def test_streams_are_recorded_as_streamed() -> None:
    fake = FakeProvider(default=Reply("a b"))
    with Client(provider_factory=fake.factory) as client:
        list(client.stream("gpt-5.5", "hi"))
    assert [seen.streamed for seen in fake.requests] == [True]
