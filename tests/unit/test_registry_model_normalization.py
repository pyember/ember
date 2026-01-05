import asyncio

from ember.models import ModelRegistry
from ember.models.schemas import ChatResponse, UsageStats


def _make_response(model_id: str) -> ChatResponse:
    return ChatResponse(
        data="ok",
        usage=UsageStats(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model_id=model_id,
    )


def test_invoke_model_normalizes_provider_prefixed_id(monkeypatch):
    registry = ModelRegistry()

    # Stub provider with a complete() that records the model argument
    recorded = {}

    class StubProvider:
        def complete(self, prompt, model, **kwargs):  # noqa: D401 (test stub)
            recorded["prompt"] = prompt
            recorded["model"] = model
            return _make_response(model)

    # Patch get_model to return our stub
    monkeypatch.setattr(registry, "get_model", lambda _mid: StubProvider())

    resp = registry.invoke_model("openai/gpt-4", "hi")

    assert recorded["model"] == "gpt-4"
    assert resp.data == "ok"


def test_invoke_model_plain_id_passes_through(monkeypatch):
    registry = ModelRegistry()
    recorded = {}

    class StubProvider:
        def complete(self, prompt, model, **kwargs):  # noqa: D401 (test stub)
            recorded["model"] = model
            return _make_response(model)

    monkeypatch.setattr(registry, "get_model", lambda _mid: StubProvider())

    resp = registry.invoke_model("gpt-4", "hi")

    assert recorded["model"] == "gpt-4"
    assert resp.data == "ok"


def test_invoke_model_async_normalizes(monkeypatch):
    registry = ModelRegistry()
    recorded = {}

    class StubProvider:
        def complete(self, prompt, model, **kwargs):  # noqa: D401 (test stub)
            recorded["model"] = model
            return _make_response(model)

    monkeypatch.setattr(registry, "get_model", lambda _mid: StubProvider())

    async def run():
        resp = await registry.invoke_model_async("openai/gpt-4", "hi")
        return resp

    out = asyncio.run(run())
    assert recorded["model"] == "gpt-4"
    assert out.data == "ok"
