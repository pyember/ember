"""Tests for the `models()` function API."""

from unittest.mock import patch

import pytest

from ember._internal.exceptions import (
    ModelNotFoundError,
    ModelProviderError,
    ProviderAPIError,
)
from ember.api.models import ModelInvocation, Response, models


class TestModelsFunction:
    """Test the main models() function interface."""

    def test_basic_invocation(self, mock_model_response):
        """Test basic model invocation: models("gpt-4", "Hello")."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("gpt-4", "Hello")

            assert result == "Test response"
            mock_registry.invoke_model.assert_called_once_with("gpt-4", "Hello")

    def test_with_parameters(self, mock_model_response):
        """Test invocation with parameters."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("gpt-4", "Hello", temperature=0.7, max_tokens=100)

            assert result == "Test response"
            mock_registry.invoke_model.assert_called_once_with(
                "gpt-4", "Hello", temperature=0.7, max_tokens=100
            )

    def test_response_text_property(self, mock_model_response):
        """Test Response.text returns the generated content."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            response = models.response("gpt-4", "Hello")

            assert isinstance(response, Response)
            assert response.text == "Test response"
            assert str(response) == "Test response"  # __str__ returns text

    def test_response_usage_property(self, mock_model_response):
        """Test Response.usage returns token and cost info."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            response = models.response("gpt-4", "Hello")

            usage = response.usage
            assert usage["prompt_tokens"] == 10
            assert usage["completion_tokens"] == 20
            assert usage["total_tokens"] == 30
            # Cost should match actual GPT-4 pricing:
            # 10 tokens @ $0.03/1k input + 20 tokens @ $0.06/1k output = $0.0015
            assert usage["cost"] == 0.0015

    def test_response_model_id(self, mock_model_response):
        """Test Response.model_id returns the model identifier."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            response = models.response("gpt-4", "Hello")

            assert response.model_id == "gpt-4"

    def test_response_metadata_includes_latency(self, mock_model_response):
        """Response.metadata should expose latency and timestamps."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            response = models.response("gpt-4", "Hello")

            metadata = response.metadata
            assert metadata["model_id"] == "gpt-4"
            assert metadata["latency_ms"] == mock_model_response.latency_ms
            assert "created_at" in metadata

    def test_model_not_found_error(self):
        """Test error when model doesn't exist."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.side_effect = ModelNotFoundError(
                "Model 'invalid-model' not found"
            )

            with pytest.raises(ModelNotFoundError) as exc_info:
                models("invalid-model", "Hello")

            assert "invalid-model" in str(exc_info.value)

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.side_effect = ModelProviderError(
                "No API key available for model gpt-4"
            )

            with pytest.raises(ModelProviderError) as exc_info:
                models("gpt-4", "Hello")

            assert "API key" in str(exc_info.value)

    def test_provider_api_error(self):
        """Test error from provider API (rate limits, etc)."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.side_effect = ProviderAPIError("Rate limit exceeded")

            with pytest.raises(ProviderAPIError) as exc_info:
                models("gpt-4", "Hello")

            assert "Rate limit" in str(exc_info.value)

    def test_response_helper_returns_structured_response(self, mock_model_response):
        """models.response should expose the structured Response object."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            response = models.response("gpt-4", "Hello")

            assert isinstance(response, Response)
            assert response.text == "Test response"

    def test_with_metadata_returns_model_invocation(self, mock_model_response):
        """models.with_metadata returns a ModelInvocation wrapper."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models.with_metadata("gpt-4", "Hello")

            assert isinstance(result, ModelInvocation)
            assert result.text == "Test response"
            assert result.usage["total_tokens"] == 30
            assert result.model_id == "gpt-4"
            assert result.latency_ms == mock_model_response.latency_ms
            assert result.metadata["model_id"] == "gpt-4"

    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        from ember.models.schemas import ChatResponse

        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            # Response with None data
            mock_registry.invoke_model.return_value = ChatResponse(data=None)

            result = models("gpt-4", "Hello")

            assert result == ""  # Empty string for None data

    def test_response_without_usage(self):
        """Test response when usage stats are missing."""
        from ember.models.schemas import ChatResponse

        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            # Response without usage
            mock_registry.invoke_model.return_value = ChatResponse(data="Response without usage")

            response = models.response("gpt-4", "Hello")

            usage = response.usage
            assert usage["prompt_tokens"] == 0
            assert usage["completion_tokens"] == 0
            assert usage["total_tokens"] == 0
            assert usage["cost"] == 0.0

    def test_explicit_provider_format(self, mock_model_response):
        """Test using explicit provider/model format."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("openai/gpt-4", "Hello")

            assert result == "Test response"
            mock_registry.invoke_model.assert_called_once_with("openai/gpt-4", "Hello")

    def test_unicode_handling(self):
        """Test handling of unicode in prompts and responses."""
        from ember.models.schemas import ChatResponse

        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            # Unicode response
            mock_registry.invoke_model.return_value = ChatResponse(data="Hello 世界! 🌍")

            result = models("gpt-4", "你好")

            assert "世界" in result
            assert "🌍" in result
            mock_registry.invoke_model.assert_called_with("gpt-4", "你好")

    def test_discover_includes_dynamic_metadata(self, monkeypatch):
        import sys

        from ember.api.models import ModelsAPI

        models_module = sys.modules["ember.api.models"]

        dummy_info = type(
            "Info",
            (),
            {
                "provider": "dummy",
                "description": "Demo model",
                "context_window": 1024,
                "context_window_out": 256,
                "aliases": ("demo", "demo-latest"),
                "capabilities": ("stream",),
                "region_scope": ("us-east1",),
                "status": "preview",
                "hidden": False,
            },
        )()

        monkeypatch.setattr(
            models_module,
            "list_available_models",
            lambda provider=None, include_dynamic=True, refresh=False, discovery_mode=None: [
                "demo"
            ],
        )
        monkeypatch.setattr(
            models_module,
            "get_model_info",
            lambda model_id, **kwargs: dummy_info,
        )

        metadata = ModelsAPI().discover()

        assert metadata["demo"]["provider"] == "dummy"
        assert metadata["demo"]["aliases"] == list(dummy_info.aliases)
        assert metadata["demo"]["context_window_out"] == 256
