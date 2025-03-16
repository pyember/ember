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
def patch_genai() -> None:
    import google.generativeai as genai
    from unittest.mock import patch

    # Patch google's generativeai directly to avoid import path issues
    # First, save the original
    original_list_models = getattr(genai, "list_models", None)
    original_gen_model = None

    # Import deepmind_provider specifically so we can patch it directly
    try:
        # Now try to import the provider to patch
        from ember.core.registry.model.providers.deepmind import deepmind_provider

        # Get the original GenerativeModel class (if it exists)
        if hasattr(deepmind_provider, "GenerativeModel"):
            original_gen_model = deepmind_provider.GenerativeModel

        # Create a patch for list_models directly on the genai module
        genai.list_models = lambda: []

        # Create a dummy GenerativeModel and patch it on the module
        def dummy_generative_model(model_ref):
            return type(
                "DummyGenerativeModel",
                (),
                {
                    "generate_content": lambda self, prompt, generation_config, **kwargs: DummyGeminiResponse()
                },
            )()

        # Apply the patch directly
        deepmind_provider.GenerativeModel = dummy_generative_model

        yield
    finally:
        # Restore original functions if they existed
        if original_list_models:
            genai.list_models = original_list_models
        if original_gen_model:
            deepmind_provider.GenerativeModel = original_gen_model


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
