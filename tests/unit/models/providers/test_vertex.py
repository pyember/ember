"""Tests for the VertexAIProvider."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.vertex import VertexAIProvider

pytestmark = pytest.mark.usefixtures("tmp_ctx")


@pytest.fixture
def stub_vertex(monkeypatch) -> Dict[str, Any]:
    """Provide a stubbed vertexai module so tests remain hermetic."""
    tracker: Dict[str, Any] = {}

    module = types.ModuleType("vertexai")

    def init(project: str, location: str) -> None:
        tracker["project"] = project
        tracker["location"] = location

    module.init = init

    gm_module = types.ModuleType("vertexai.generative_models")

    class FakeGenerativeModel:
        def __init__(self, model_name: str) -> None:
            tracker["last_model"] = model_name

        def generate_content(self, text: str, generation_config=None):
            tracker["last_request"] = {
                "text": text,
                "generation_config": generation_config,
            }
            return SimpleNamespace(
                text="vertex-response",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=2,
                    candidates_token_count=3,
                    total_token_count=5,
                ),
            )

    gm_module.GenerativeModel = FakeGenerativeModel
    module.generative_models = gm_module

    monkeypatch.setitem(sys.modules, "vertexai", module)
    monkeypatch.setitem(sys.modules, "vertexai.generative_models", gm_module)

    return tracker


def test_vertex_provider_generates_content(stub_vertex, tmp_ctx):
    """Happy path flows through GenerativeModel.generate_content."""
    tmp_ctx.set_config("providers.vertex.project", "proj-123")
    tmp_ctx.set_config("providers.vertex.location", "us-central2")

    provider = VertexAIProvider()
    response = provider.complete(
        "ping",
        "gemini-1.5-pro",
        temperature=0.3,
        max_tokens=256,
        stop=["END"],
        system="system-msg",
    )

    tracker = stub_vertex
    assert tracker["project"] == "proj-123"
    assert tracker["location"] == "us-central2"
    assert tracker["last_model"] == "gemini-1.5-pro"

    last_request = tracker["last_request"]
    assert "system-msg" in last_request["text"]
    assert last_request["text"].endswith("ping")

    config = last_request["generation_config"]
    assert config["temperature"] == 0.3
    assert config["max_output_tokens"] == 256
    assert config["stop_sequences"] == ["END"]

    assert response.data == "vertex-response"
    assert response.usage.total_tokens == 5


def test_vertex_provider_requires_project(stub_vertex, tmp_ctx):
    """Project configuration is mandatory."""
    tmp_ctx.set_config("providers.vertex.location", "europe-west4")

    with pytest.raises(ValueError, match="providers\\.vertex\\.project"):
        VertexAIProvider()


def test_vertex_provider_rejects_stream(stub_vertex, tmp_ctx):
    """Streaming is not yet supported."""
    tmp_ctx.set_config("providers.vertex.project", "proj-456")

    provider = VertexAIProvider()
    with pytest.raises(ProviderAPIError, match="Streaming responses are not implemented"):
        provider.complete("prompt", "gemini-1.5-flash", stream=True)
