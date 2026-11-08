"""`ember.models`: the public signature, the active client and binding."""

import doctest
import importlib
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

import ember
import ember.llm
from ember.errors import ConfigError
from ember.llm import Client, using
from ember.llm.api import Models
from ember.llm.fake import FakeProvider, Reply

OPTIONS = [
    "system",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_output_tokens",
    "stop",
    "seed",
    "reasoning_effort",
    "output_schema",
    "extensions",
]


def test_ember_models_is_the_one_models_object() -> None:
    assert ember.models is ember.llm.models
    assert type(ember.models) is Models
    assert repr(ember.models) == "ember.models"


@pytest.mark.parametrize(
    ("function", "positional"),
    [
        (Models.__call__, ["self", "model", "prompt"]),
        (Models.stream, ["self", "model", "prompt"]),
        (Models.bind, ["self", "model"]),
        (Client.complete, ["self", "model", "prompt"]),
        (Client.stream, ["self", "model", "prompt"]),
        (Client.bind, ["self", "model"]),
    ],
)
def test_every_call_surface_takes_exactly_the_documented_arguments(
    function: Callable[..., object], positional: list[str]
) -> None:
    parameters = inspect.signature(function).parameters.values()
    assert [p.name for p in parameters] == positional + OPTIONS
    kinds = {p.name: p.kind for p in parameters}
    assert all(kinds[name] is inspect.Parameter.KEYWORD_ONLY for name in OPTIONS)
    assert not any(
        p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        for p in parameters
    )


def test_using_routes_calls_and_restores_the_outer_client() -> None:
    outer_fake, inner_fake = (
        FakeProvider(default=Reply("outer")),
        FakeProvider(default=Reply("inner")),
    )
    with (
        Client(provider_factory=outer_fake.factory) as outer,
        Client(provider_factory=inner_fake.factory) as inner,
    ):
        with using(outer):
            assert ember.models("gpt-5.5", "hi").text == "outer"
            with using(inner):
                assert ember.models("gpt-5.5", "hi").text == "inner"
            assert ember.models("gpt-5.5", "hi").text == "outer"
    assert (len(outer_fake.requests), len(inner_fake.requests)) == (2, 1)


def test_models_bind_uses_the_client_active_when_called() -> None:
    bound = ember.models.bind("gpt-5.5", system="Be brief.", max_output_tokens=50)
    first, second = FakeProvider(), FakeProvider()
    with Client(provider_factory=first.factory) as a, Client(provider_factory=second.factory) as b:
        with using(a):
            bound("one")
        with using(b):
            bound("two")
    [seen_first], [seen_second] = first.requests, second.requests
    assert (seen_first.request.system, seen_first.request.sampling.max_output_tokens) == (
        "Be brief.",
        50,
    )
    assert seen_second.request.messages[0].parts[0].text == "two"  # type: ignore[union-attr]


def test_client_bind_stays_on_its_own_client() -> None:
    own, other = FakeProvider(), FakeProvider()
    with (
        Client(provider_factory=own.factory) as mine,
        Client(provider_factory=other.factory) as theirs,
    ):
        bound = mine.bind("claude-haiku-5-5")
        with using(theirs):
            bound("hi")
    assert (len(own.requests), len(other.requests)) == (1, 0)


def test_binding_reads_nothing_until_called(isolated_config: Path) -> None:
    isolated_config.write_text("not: [valid\n")
    bound = ember.models.bind("gpt-5.5")
    with Client(provider_factory=FakeProvider().factory) as client, using(client):
        with pytest.raises(ConfigError):
            bound("hi")


@pytest.mark.parametrize(
    "module", ["ember.llm.api", "ember.llm.catalog", "ember.llm.client", "ember.llm.fake"]
)
def test_docstring_examples_run(module: str) -> None:
    results = doctest.testmod(importlib.import_module(module))
    assert results.failed == 0
    assert results.attempted > 0
