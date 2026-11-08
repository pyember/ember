"""`Client.check` raises exactly what a call would raise before sending, and sends nothing."""

import pytest

from ember import context
from ember.errors import ConfigError, ErrorKind, ModelError
from ember.llm import Client, catalog
from ember.llm.fake import FakeProvider
from ember.llm.ref import ModelRef
from ember.llm.request import Sampling


def test_check_resolves_an_alias_and_sends_nothing(client: Client, fake: FakeProvider) -> None:
    spec = client.check("gpt-5.5", sampling=Sampling(max_output_tokens=100))
    assert spec.ref == ModelRef("openai", "gpt-5.5")
    assert (fake.requests, fake.built) == ((), ())


@pytest.mark.parametrize(
    ("model", "sampling", "error"),
    [
        ("not-a-model", Sampling(), ModelError),
        ("nowhere/model", Sampling(), ModelError),
        ("openai/gpt-5.5", Sampling(temperature=0.5), ValueError),
        ("openai/gpt-5.5", Sampling(reasoning_effort="enormous"), ValueError),
        ("openai/gpt-5.5", Sampling(max_output_tokens=10_000_000), ValueError),
    ],
)
def test_check_raises_what_the_call_raises(
    client: Client,
    fake: FakeProvider,
    model: str,
    sampling: Sampling,
    error: type[Exception],
) -> None:
    with pytest.raises(error) as checked:
        client.check(model, sampling=sampling)
    with pytest.raises(error) as called:
        client.complete(
            model,
            "hi",
            temperature=sampling.temperature,
            max_output_tokens=sampling.max_output_tokens,
            reasoning_effort=sampling.reasoning_effort,
        )
    assert str(checked.value) == str(called.value)
    assert fake.requests == ()


def test_check_applies_the_price_policy(client: Client, fake: FakeProvider) -> None:
    unpriced = next(spec for spec in catalog.default() if spec.rates is None)
    with context.override(require_price=True), pytest.raises(ModelError) as raised:
        client.check(unpriced.ref)
    assert raised.value.kind is ErrorKind.UNPRICED
    assert fake.requests == ()


def test_check_needs_a_model_or_a_default(client: Client) -> None:
    with pytest.raises(ConfigError, match="default_model"):
        client.check(None)
