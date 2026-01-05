from types import SimpleNamespace

import pytest

from ember.models.providers.google import GoogleDiscoveryAdapter


def _make_client_module(model):
    class _DummyModels:
        def list(self, parent=None):  # noqa: D401 - mimic SDK signature
            return [model]

    class _DummyClient:
        def __init__(self, api_key):
            self.models = _DummyModels()

    return SimpleNamespace(Client=_DummyClient)


def _make_legacy_module(model):
    def list_models(**kwargs):  # noqa: D401 - mimic SDK signature
        return [model]

    def configure(**kwargs):  # noqa: D401 - mimic SDK signature
        return None

    return SimpleNamespace(list_models=list_models, configure=configure)


class _DummyModel:
    def __init__(self):
        self.name = "models/gemini-2.5-pro"
        self.display_name = "Gemini 2.5 Pro"
        self.description = "Latest pro tier"
        self.input_token_limit = 1_000_000
        self.output_token_limit = 1_000_000
        self.supported_actions = ["generateContent", "embedContent"]

    def to_dict(self):
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
        }


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda model: _make_client_module(model),
        lambda model: _make_legacy_module(model),
    ],
)
def test_google_discovery_adapter_returns_discovered_models(module_factory):
    model = _DummyModel()
    module = module_factory(model)

    adapter = GoogleDiscoveryAdapter(
        api_key_resolver=lambda: "test-key",
        module_loader=lambda: module,
    )

    discovered = list(adapter.list_models())
    assert len(discovered) == 1
    entry = discovered[0]
    assert entry.provider == "google"
    assert entry.id == "gemini-2.5-pro"
    assert entry.display_name == "Gemini 2.5 Pro"
    assert entry.context_window_in == 1_000_000
    assert set(entry.capabilities) == {"generateContent", "embedContent"}
    assert entry.raw_payload == {
        "name": "models/gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "description": "Latest pro tier",
    }


def test_google_discovery_adapter_without_api_key_returns_empty():
    adapter = GoogleDiscoveryAdapter(api_key_resolver=lambda: None, module_loader=lambda: None)
    assert list(adapter.list_models()) == []
