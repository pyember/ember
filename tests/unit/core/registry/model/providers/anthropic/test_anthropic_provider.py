#!/usr/bin/env python3
"""Unit tests for the AnthropicModel provider implementation.
"""

import pytest
from typing import Any
import anthropic

from ember.core.registry.model.providers.anthropic.anthropic_provider import (
    AnthropicModel,
    AnthropicChatParameters,
)
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatResponse,
    ChatRequest,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit


class DummyAnthropicResponse:
    def __init__(self) -> None:
        self.completion = " Dummy anthropic response. "


def create_dummy_anthropic_model_info() -> ModelInfo:
    return ModelInfo(
        id="anthropic:claude-3.5-sonnet",
        name="claude-3.5-sonnet",
        cost=ModelCost(input_cost_per_thousand=3, output_cost_per_thousand=15),
        rate_limit=RateLimit(tokens_per_minute=300000, requests_per_minute=2000),
        provider=ProviderInfo(name="Anthropic", default_api_key="dummy_anthropic_key"),
        api_key="dummy_anthropic_key",
    )


@pytest.fixture(autouse=True)
def patch_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyCompletions:
        def create(self, **kwargs: Any) -> Any:
            return DummyAnthropicResponse()

    class DummyClient:
        completions = DummyCompletions()

    monkeypatch.setattr(anthropic, "Client", lambda api_key: DummyClient())


def test_anthropic_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that AnthropicModel.forward returns a ChatResponse with correct data."""
    dummy_info = create_dummy_anthropic_model_info()
    model = AnthropicModel(dummy_info)
    request = ChatRequest(prompt="Hello Anthropic", temperature=0.9, max_tokens=100)
    response = model.forward(request)
    assert isinstance(response, ChatResponse)
    assert "Dummy anthropic response." in response.data.strip()


def test_anthropic_parameters() -> None:
    """Test that AnthropicChatParameters enforces defaults and converts parameters."""
    params = AnthropicChatParameters(prompt="Test", max_tokens=None)
    kwargs = params.to_anthropic_kwargs()
    # Default max_tokens should be 768 if None provided.
    assert kwargs["max_tokens_to_sample"] == 768
