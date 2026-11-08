"""ModelCall: one call per input, no default model, no repr-as-prompt."""

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from ember.llm.ref import ModelRef
from ember.llm.request import Message, Sampling
from ember.operators import ModelCall


def test_string_input_is_sent_as_is_with_settings(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: f"echo {prompt}")
    call = ModelCall(
        "openai/gpt-5",
        system="be terse",
        sampling=Sampling(temperature=0.2, max_output_tokens=64, seed=7),
    )
    result = call("hello")
    assert result.text == "echo hello"
    assert result.model == ModelRef("openai", "gpt-5")
    [sent] = fake.calls
    assert (sent.model, sent.prompt, sent.system) == ("openai/gpt-5", "hello", "be terse")
    assert (sent.temperature, sent.max_output_tokens, sent.seed) == (0.2, 64, 7)


def test_template_substitutes_the_input_without_interpreting_it(
    fake_models: Callable[..., Any],
) -> None:
    fake = fake_models(lambda model, prompt: "ok")
    ModelCall("openai/gpt-5", template="Q: {input}\nA:")("what is {x}?")
    assert fake.calls[0].prompt == "Q: what is {x}?\nA:"


def test_messages_are_sent_as_a_tuple(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: "ok")
    history = [Message.user("hi"), Message.assistant("hello"), Message.user("bye")]
    ModelCall("anthropic/claude-sonnet-4-5")(history)
    assert fake.calls[0].prompt == tuple(history)


@pytest.mark.parametrize(
    "value",
    [
        {"question": "capital of France?", "context": "geo"},
        None,
        ["Paris", "Lyon"],
        42,
    ],
)
def test_anything_but_text_or_messages_raises_before_sending(
    fake_models: Callable[..., Any], value: object
) -> None:
    fake = fake_models(lambda model, prompt: "should not be called")
    with pytest.raises(TypeError, match="string or a sequence of Message"):
        ModelCall("openai/gpt-5")(value)  # type: ignore[arg-type]
    assert fake.calls == []


def test_a_template_needs_text_input(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: "ok")
    with pytest.raises(TypeError, match="with a template takes a string"):
        ModelCall("openai/gpt-5", template="{input}")([Message.user("hi")])


def test_there_is_no_default_model() -> None:
    with pytest.raises(TypeError):
        ModelCall()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="needs a model"):
        ModelCall("  ")


@pytest.mark.parametrize(
    ("template", "found"),
    [
        ("no field at all", "must contain {input}"),
        ("{question}", "found {question}"),
        ("{input!r}", "found {input!r}"),
        ("{input:>10}", "found {input:>10}"),
        ("{}", "found {}"),
        ("{input} {", "Single '{'"),
    ],
)
def test_templates_are_checked_at_construction(template: str, found: str) -> None:
    with pytest.raises(ValueError, match=found.replace("{", r"\{").replace("}", r"\}")):
        ModelCall("openai/gpt-5", template=template)


def test_replace_changes_settings_and_revalidates() -> None:
    call = ModelCall("openai/gpt-5")
    hotter = dataclasses.replace(call, sampling=Sampling(temperature=1.0))
    assert hotter.sampling.temperature == 1.0 and call.sampling.temperature is None
    with pytest.raises(ValueError):
        dataclasses.replace(call, template="{oops}")
