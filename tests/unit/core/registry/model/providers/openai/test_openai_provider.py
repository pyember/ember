#!/usr/bin/env python3
"""Unit tests for the OpenAIModel provider implementation.
"""

import pytest
from typing import Any

from ember.core.registry.model.providers.openai.openai_provider import (
    OpenAIModel,
    OpenAIChatParameters,
)
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatResponse,
    ChatRequest,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit


class DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyChoice:
    def __init__(self, message_content: str) -> None:
        self.message = DummyMessage(message_content)


class DummyOpenAIResponse:
    def __init__(self) -> None:
        self.choices = [DummyChoice("Test response.")]
        self.usage = type(
            "Usage",
            (),
            {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60},
        )


def create_dummy_model_info() -> ModelInfo:
    return ModelInfo(
        id="openai:gpt-4o",
        name="gpt-4o",
        cost=ModelCost(input_cost_per_thousand=5000, output_cost_per_thousand=15000),
        rate_limit=RateLimit(tokens_per_minute=10000000, requests_per_minute=1500),
        provider=ProviderInfo(name="OpenAI", default_api_key="dummy_openai_key"),
        api_key="dummy_openai_key",
    )


class DummyOpenAIClient:
    class DummyCompletions:
        @staticmethod
        def create(**kwargs: Any) -> Any:
            return DummyOpenAIResponse()

    chat = type("DummyChat", (), {"completions": DummyCompletions()})()


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    monkeypatch.setattr(openai, "chat", DummyOpenAIClient().chat)


def test_openai_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenAIModel.forward returns a valid ChatResponse."""
    dummy_info = create_dummy_model_info()
    model = OpenAIModel(dummy_info)
    # Patch client.chat.completions.create to return our dummy response.
    monkeypatch.setattr(
        model.client.chat.completions, "create", lambda **kwargs: DummyOpenAIResponse()
    )
    request = ChatRequest(prompt="Hello OpenAI", temperature=0.7, max_tokens=100)
    response = model.forward(request)
    assert isinstance(response, ChatResponse)
    assert "Test response." in response.data
    usage = response.usage
    assert usage.total_tokens == 100


def test_openai_parameters() -> None:
    """Test that OpenAIChatParameters converts prompt to messages properly."""
    params = OpenAIChatParameters(
        prompt="Hello", context="System message", temperature=0.5, max_tokens=None
    )
    kwargs = params.to_openai_kwargs()
    assert "messages" in kwargs
    messages = kwargs["messages"]
    assert any(msg["role"] == "system" for msg in messages)
    assert any(msg["role"] == "user" for msg in messages)
