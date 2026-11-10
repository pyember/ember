"""`Client.check` raises exactly what a call would raise before sending, and sends nothing."""

import httpx
import pytest

from ember import context
from ember.context import Context, ProviderSettings
from ember.errors import ConfigError, ErrorKind, ModelError
from ember.llm import Client, catalog, providers
from ember.llm.fake import FakeProvider
from ember.llm.provider import Provider
from ember.llm.ref import ModelRef
from ember.llm.request import Sampling


def test_check_resolves_an_alias_and_sends_nothing(client: Client, fake: FakeProvider) -> None:
    spec = client.check("gpt-5.5", sampling=Sampling(max_output_tokens=100))
    assert spec.ref == ModelRef("openai", "gpt-5.5")
    assert fake.requests == ()
    assert [settings.name for settings in fake.built] == ["openai"]


def test_check_builds_only_what_the_next_call_reuses(client: Client, fake: FakeProvider) -> None:
    client.check("openai/gpt-5.5")
    client.check("openai/gpt-5.5")
    client.complete("openai/gpt-5.5", "hi").result()
    assert len(fake.built) == 1
    assert len(fake.requests) == 1


def test_a_missing_key_raises_from_check_exactly_as_from_a_call() -> None:
    with Client() as real:
        with pytest.raises(ConfigError) as checked:
            real.check("openai/gpt-5.5")
        with pytest.raises(ConfigError) as called:
            real.complete("openai/gpt-5.5", "hi")
    assert str(checked.value) == str(called.value)
    assert "has no API key" in str(checked.value)


def test_a_provider_whose_key_is_optional_passes_check_without_one() -> None:
    local = {"kind": "openai_compatible", "base_url": "http://127.0.0.1:9/v1"}
    with context.override(providers={"local": local}), Client() as real:
        assert real.check("local/any-model").ref == ModelRef("local", "any-model")


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


def test_a_missing_base_url_raises_from_check_exactly_as_from_a_call() -> None:
    no_url = {"kind": "openai_compatible"}
    with context.override(providers={"local": no_url}), Client() as real:
        with pytest.raises(ConfigError) as checked:
            real.check("local/any-model")
        with pytest.raises(ConfigError) as called:
            real.complete("local/any-model", "hi")
    assert str(checked.value) == str(called.value)
    assert "needs base_url" in str(checked.value)


def test_the_provider_check_built_is_closed_with_the_client(fake: FakeProvider) -> None:
    with Client(provider_factory=fake.factory) as client:
        client.check("openai/gpt-5.5")
        assert (len(fake.built), fake.closes) == (1, 0)
    assert fake.closes == 1


def test_check_with_the_real_factory_opens_no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(500)

    def factory(settings: ProviderSettings, ctx: Context) -> Provider:
        return providers.build(settings, ctx, transport=httpx.MockTransport(handler))

    with Client(provider_factory=factory) as real:
        assert real.check("openai/gpt-5.5").ref == ModelRef("openai", "gpt-5.5")
    assert sent == []
