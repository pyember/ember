import json
from typing import Any, Dict, Iterable, Optional

import pytest

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.ollama import OllamaProvider


class _FakeHTTPErrorResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakeResponse:
    def __init__(self, data: Dict[str, Any], status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            import httpx

            raise httpx.HTTPStatusError(
                "error",
                request=None,
                response=_FakeHTTPErrorResponse(self.status_code, self.text),
            )

    def json(self) -> Dict[str, Any]:
        return dict(self._data)


class _FakeStream:
    def __init__(self, lines: Iterable[str]):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.last_url: Optional[str] = None
        self.last_json: Optional[Dict[str, Any]] = None
        # Configurable hooks
        self._post_response: Optional[_FakeResponse] = None
        self._post_raises: Optional[BaseException] = None
        self._stream_lines: Optional[list[str]] = None

    # Context manager behavior
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    # API
    def post(self, url: str, json: Dict[str, Any]):
        self.last_url = url
        self.last_json = json
        if self._post_raises:
            raise self._post_raises
        resp = self._post_response or _FakeResponse({"ok": True})
        resp.raise_for_status()
        return resp

    def stream(self, method: str, url: str, json: Dict[str, Any]):
        assert method == "POST"
        self.last_url = url
        self.last_json = json
        lines = self._stream_lines or []
        return _FakeStream(lines)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("EMBER_OLLAMA_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("EMBER_OLLAMA_TOKEN_CHAR_RATIO", raising=False)
    yield


def _patch_client(monkeypatch, fake: _FakeClient):
    # Patch the Client used inside the provider module
    import importlib

    mod = importlib.import_module("ember.models.providers.ollama")
    monkeypatch.setattr(mod.httpx, "Client", lambda timeout=30.0: fake)
    return mod


def test_generate_endpoint_body_and_mapping(monkeypatch):
    fake = _FakeClient()
    fake._post_response = _FakeResponse({"response": "Hi!"}, status_code=200)
    _patch_client(monkeypatch, fake)

    prov = OllamaProvider()
    resp = prov.complete(
        "Hello",
        model="llama3.1:8b",
        temperature=0.2,
        top_p=0.9,
        max_tokens=42,
        stop=["\n\n"],
    )

    assert resp.data.startswith("Hi")
    assert resp.model_id == "llama3.1:8b"
    assert resp.usage.total_tokens > 0

    assert fake.last_url and fake.last_url.endswith("/api/generate")
    assert fake.last_json is not None
    body = fake.last_json
    assert body["model"] == "llama3.1:8b"
    assert body["prompt"] == "Hello"
    assert body["options"]["temperature"] == 0.2
    assert body["options"]["top_p"] == 0.9
    assert body["options"]["num_predict"] == 42
    assert body["options"]["stop"] == ["\n\n"]


def test_chat_endpoint_when_system_present(monkeypatch):
    fake = _FakeClient()
    fake._post_response = _FakeResponse({"message": {"role": "assistant", "content": "OK"}}, 200)
    _patch_client(monkeypatch, fake)

    prov = OllamaProvider()
    resp = prov.complete("User question", model="llama3.1:8b", context="You are helpful.")
    assert resp.data == "OK"
    assert fake.last_url and fake.last_url.endswith("/api/chat")
    body = fake.last_json or {}
    assert body.get("messages") and body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


def test_connection_error_normalized(monkeypatch):
    import httpx

    fake = _FakeClient()
    fake._post_raises = httpx.ConnectError("refused")
    _patch_client(monkeypatch, fake)

    prov = OllamaProvider()
    with pytest.raises(ProviderAPIError) as ei:
        prov.complete("hi", model="llama3.1:8b")
    msg = str(ei.value)
    assert "Ollama" in msg or "reach" in msg


def test_http_404_model_hint(monkeypatch):
    fake = _FakeClient()
    # Response with 404 to trigger hint
    fake._post_response = _FakeResponse({"error": "not found"}, status_code=404)
    _patch_client(monkeypatch, fake)

    prov = OllamaProvider()
    with pytest.raises(ProviderAPIError) as ei:
        prov.complete("hi", model="missing-model")
    assert "ollama run" in str(ei.value).lower() or "hint" in str(ei.value).lower()


def test_streaming_generate_is_aggregated(monkeypatch):
    fake = _FakeClient()
    fake._stream_lines = [
        json.dumps({"response": "Hello"}),
        json.dumps({"response": " world"}),
        json.dumps({"done": True}),
    ]
    _patch_client(monkeypatch, fake)

    prov = OllamaProvider()
    resp = prov.complete("hi", model="llama3.1:8b", stream=True)
    assert resp.data == "Hello world"
    assert fake.last_json and fake.last_json.get("stream") is True


def test_base_url_config_override(monkeypatch):
    fake = _FakeClient()
    fake._post_response = _FakeResponse({"response": "ok"}, 200)
    _patch_client(monkeypatch, fake)

    from ember._internal.context.runtime import EmberContext

    with EmberContext.current().create_child(
        providers={"ollama": {"base_url": "http://127.0.0.1:9999"}}
    ):
        prov = OllamaProvider()
        prov.complete("hello", model="llama3.1:8b")
    assert fake.last_url.startswith("http://127.0.0.1:9999")
