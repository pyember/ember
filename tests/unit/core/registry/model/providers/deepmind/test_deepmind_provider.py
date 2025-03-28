#!/usr/bin/env python3
"""Unit tests for the GeminiModel (Deepmind provider) implementation."""

import pytest

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.providers.deepmind.deepmind_provider import (
    GeminiChatParameters,
    GeminiModel,
)


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
        class DummyGenerativeModel:
            def __init__(self, model_ref):
                self.model_ref = model_ref

            def generate_content(self, *, contents, generation_config, **kwargs):
                return DummyGeminiResponse()

        deepmind_provider.GenerativeModel = DummyGenerativeModel

        # Apply the patch directly
        # This line was redundant since we already assigned DummyGenerativeModel above

        yield
    finally:
        # Restore original functions if they existed
        if original_list_models:
            genai.list_models = original_list_models
        if original_gen_model:
            deepmind_provider.GenerativeModel = original_gen_model


def test_deepmind_forward() -> None:
    """Test that GeminiModel.forward returns a valid ChatResponse."""
    # Fix the type checking issue by directly examining the response content
    import inspect

    # Get the module where ChatResponse is defined
    response_module = inspect.getmodule(ChatResponse)

    dummy_info = create_dummy_deepmind_model_info()
    model = GeminiModel(dummy_info)
    request = ChatRequest(prompt="Hello Gemini", temperature=0.7, max_tokens=100)
    response = model.forward(request)

    # Verify it's a ChatResponse by checking structure and behavior,
    # not by using isinstance which can be affected by module loading
    assert response.__class__.__name__ == "ChatResponse"
    assert hasattr(response, "data")
    assert hasattr(response, "raw_output")
    assert hasattr(response, "usage")

    # Verify the actual content/behavior
    assert "Gemini response text" in response.data
    usage = response.usage
    assert usage.total_tokens == 70


def test_gemini_parameters() -> None:
    """Test that GeminiChatParameters enforces defaults and converts parameters."""
    params = GeminiChatParameters(prompt="Test", max_tokens=None)
    kwargs = params.to_gemini_kwargs()
    # Default max_tokens should be 512.
    assert kwargs["generation_config"]["max_output_tokens"] == 512
