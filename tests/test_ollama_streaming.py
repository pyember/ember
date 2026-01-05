import json

from ember.models.providers.ollama import OllamaProvider


class _FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for ln in self._lines:
            yield ln


class _FakeClient:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, method, url, json):
        assert method == "POST"
        return _FakeStream(self._lines)


def test_stream_complete_yields_and_returns(monkeypatch):
    # Simulate generate stream
    lines = [
        json.dumps({"response": "Hel"}),
        json.dumps({"response": "lo"}),
        json.dumps({"done": True}),
    ]

    import importlib

    mod = importlib.import_module("ember.models.providers.ollama")

    monkeypatch.setattr(mod.httpx, "Client", lambda timeout=30.0: _FakeClient(lines))

    prov = OllamaProvider()
    gen = prov.stream_complete("hi", "llama3.1:8b")

    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as e:
        final = e.value
        assert "".join(chunks) == "Hello"
        assert final.data == "Hello"
        assert final.usage and final.usage.total_tokens > 0
