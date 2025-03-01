#!/usr/bin/env python3
"""Unit tests for the GeminiModel (Deepmind provider) implementation.
"""

import pytest

from ember.core.registry.model.providers.deepmind.deepmind_provider import (
    GeminiModel,
    GeminiChatParameters,
)
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatResponse,
    ChatRequest,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit


class DummyGeminiResponse:
    def __init__(self) -> None:
        self.text = "Gemini response text"
        self.usage_metadata = type(
            "UsageMeta",
            (),
            {
                "prompt_token_count": 50,
                "candidates_token_count": 20,
                "total_token_count": 70,
            },
        )


def create_dummy_deepmind_model_info() -> ModelInfo:
    return ModelInfo(
        id="deepmind:gemini-1.5-pro",
        name="gemini-1.5-pro",
        cost=ModelCost(input_cost_per_thousand=3500, output_cost_per_thousand=10500),
        rate_limit=RateLimit(tokens_per_minute=1000000, requests_per_minute=1000),
        provider=ProviderInfo(name="Google", default_api_key="dummy_google_key"),
        api_key="dummy_google_key",
    )


@pytest.fixture(autouse=True)
def patch_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.generativeai as genai

    # Patch list_models to return an empty list (we don't need it here).
    monkeypatch.setattr(genai, "list_models", lambda: [])
    # Patch GenerativeModel to be our dummy.
    monkeypatch.setattr(
        "ember.core.registry.model.providers.deepmind.deepmind_provider.GenerativeModel",
        lambda model_ref: type(
            "DummyGenerativeModel",
            (),
            {
                "generate_content": lambda self, prompt, generation_config, **kwargs: DummyGeminiResponse()
            },
        )(),
    )


def test_deepmind_forward() -> None:
    """Test that GeminiModel.forward returns a valid ChatResponse."""
    dummy_info = create_dummy_deepmind_model_info()
    model = GeminiModel(dummy_info)
    request = ChatRequest(prompt="Hello Gemini", temperature=0.7, max_tokens=100)
    response = model.forward(request)
    assert isinstance(response, ChatResponse)
    assert "Gemini response text" in response.data
    usage = response.usage
    assert usage.total_tokens == 70


def test_gemini_parameters() -> None:
    """Test that GeminiChatParameters enforces defaults and converts parameters."""
    params = GeminiChatParameters(prompt="Test", max_tokens=None)
    kwargs = params.to_gemini_kwargs()
    # Default max_tokens should be 512.
    assert kwargs["generation_config"]["max_output_tokens"] == 512
