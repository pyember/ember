"""Tests for the BedrockConverseProvider."""

from __future__ import annotations

import sys
import types
from typing import Any, Dict

import pytest

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.bedrock import BedrockConverseProvider

pytestmark = pytest.mark.usefixtures("tmp_ctx")


@pytest.fixture
def stub_bedrock(monkeypatch) -> Dict[str, Any]:
    """Stub boto3 client for deterministic testing."""
    tracker: Dict[str, Any] = {}

    class FakeClient:
        def __init__(self, service_name: str, region_name: str):
            tracker["service"] = service_name
            tracker["region"] = region_name

        def converse(self, **payload):
            tracker["payload"] = payload
            return {
                "output": {
                    "message": {
                        "content": [
                            {"text": "bedrock says hi"},
                        ]
                    }
                },
                "usage": {
                    "inputTokens": 7,
                    "outputTokens": 11,
                    "totalTokens": 18,
                },
            }

    module = types.ModuleType("boto3")

    def client(name: str, region_name: str | None = None):
        return FakeClient(name, region_name or "")

    module.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", module)

    return tracker


def test_bedrock_completion_happy_path(stub_bedrock, tmp_ctx):
    """Provider issues converse calls and normalizes response."""
    tmp_ctx.set_config("providers.bedrock.region", "us-west-2")

    provider = BedrockConverseProvider()
    response = provider.complete(
        "ping",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        system="system prompt",
        context=["hint A", "hint B"],
        temperature=0.1,
        max_tokens=1024,
        stop=["END"],
    )

    tracker = stub_bedrock
    assert tracker["service"] == "bedrock-runtime"
    assert tracker["region"] == "us-west-2"

    payload = tracker["payload"]
    assert payload["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0]["text"] == "ping"

    system_prompts = payload["system"]
    assert system_prompts[0]["text"] == "system prompt"
    assert system_prompts[1]["text"] == "hint A"
    assert system_prompts[2]["text"] == "hint B"

    inference = payload["inferenceConfig"]
    assert inference["temperature"] == 0.1
    assert inference["maxTokens"] == 1024
    assert inference["stopSequences"] == ["END"]

    assert response.data == "bedrock says hi"
    assert response.usage.total_tokens == 18
    assert response.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_bedrock_rejects_stream(stub_bedrock, tmp_ctx):
    """Streaming not yet supported."""
    tmp_ctx.set_config("providers.bedrock.region", "us-east-1")

    provider = BedrockConverseProvider()
    with pytest.raises(ProviderAPIError, match="Streaming is not implemented"):
        provider.complete("prompt", "model", stream=True)


def test_bedrock_stop_requires_sequence(stub_bedrock, tmp_ctx):
    """Invalid stop parameter surfaces a ProviderAPIError."""
    tmp_ctx.set_config("providers.bedrock.region", "us-east-1")

    provider = BedrockConverseProvider()
    with pytest.raises(ProviderAPIError, match="stop must be a string or sequence"):
        provider.complete("prompt", "model", stop=123)
