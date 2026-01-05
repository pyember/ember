import types

from ember.models.discovery.types import DiscoveredModel
from ember.models.providers.anthropic import AnthropicDiscoveryAdapter
from ember.models.providers.openai import OpenAIDiscoveryAdapter


class _DummyOpenAIModels:
    def __init__(self, responses):
        self._responses = responses

    def list(self):
        return types.SimpleNamespace(data=self._responses)


class _DummyOpenAIClient:
    def __init__(self, responses):
        self.models = _DummyOpenAIModels(responses)


class _DummyAnthropicPage:
    def __init__(self, responses):
        self.data = responses

    def iter_pages(self):
        yield self


class _DummyAnthropicModels:
    def __init__(self, responses):
        self._responses = responses

    def list(self, limit=100):  # noqa: D401 - matches SDK signature
        return _DummyAnthropicPage(self._responses)


class _DummyAnthropicClient:
    def __init__(self, responses):
        self.models = _DummyAnthropicModels(responses)


def test_openai_discovery_adapter_maps_basic_fields():
    responses = [types.SimpleNamespace(id="gpt-test", display_name="GPT Test")]
    adapter = OpenAIDiscoveryAdapter(
        api_key_resolver=lambda: "sk-test",
        client_factory=lambda key: _DummyOpenAIClient(responses),
    )

    result = list(adapter.list_models())

    assert result == [DiscoveredModel(provider="openai", id="gpt-test", display_name="GPT Test")]


def test_anthropic_discovery_adapter_maps_payload():
    """Anthropic adapter extracts id and display_name from response objects.

    Note: SimpleNamespace doesn't have model_dump() or to_dict(), so
    raw_payload is empty. Real Anthropic SDK responses would include payload.
    """
    responses = [types.SimpleNamespace(id="claude-test", display_name="Claude Test")]
    adapter = AnthropicDiscoveryAdapter(
        api_key_resolver=lambda: "sk-test",
        client_factory=lambda key: _DummyAnthropicClient(responses),
    )

    result = list(adapter.list_models())

    assert result == [
        DiscoveredModel(
            provider="anthropic",
            id="claude-test",
            display_name="Claude Test",
            raw_payload={},  # SimpleNamespace doesn't serialize to dict
        )
    ]
