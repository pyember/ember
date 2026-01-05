import pytest

from ember.models import ModelRegistry
from ember.models.providers import resolve_model_id


@pytest.fixture(autouse=True)
def enable_local_alias(monkeypatch):
    monkeypatch.setenv("EMBER_LOCAL_ALIAS", "1")
    yield


def test_resolve_model_id_local_alias():
    provider, name = resolve_model_id("local/llama3.1:8b")
    assert provider == "ollama"
    assert name == "llama3.1:8b"


def test_registry_with_local_alias(monkeypatch):
    # Patch httpx used by Ollama provider to avoid network
    class _FakeResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return dict(self._data)

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            # Minimal response shape for generate endpoint
            return _FakeResp({"response": "ok"})

    import importlib

    mod = importlib.import_module("ember.models.providers.ollama")
    monkeypatch.setattr(mod.httpx, "Client", lambda timeout=30.0: _FakeClient())

    from ember._internal.context.runtime import EmberContext

    with EmberContext.current().create_child(
        providers={"ollama": {"base_url": "http://127.0.0.1:11434"}}
    ):
        registry = ModelRegistry()
        resp = registry.invoke_model("local/llama3.1:8b", "hi")
    assert resp.data == "ok"
