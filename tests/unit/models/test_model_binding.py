"""Tests for model.instance() bindings."""

from unittest.mock import Mock, patch

import pytest

from ember._internal.exceptions import ModelNotFoundError
from ember.api.models import ModelBinding, ModelInvocation, Response, models


class TestModelBinding:
    """Test the ModelBinding class for preset configurations."""

    def test_create_binding(self):
        """Test creating a model binding with parameters."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()  # Model exists

            binding = models.instance("gpt-4", temperature=0.7, max_tokens=100)

            assert isinstance(binding, ModelBinding)
            assert binding.model_id == "gpt-4"
            assert binding.params == {"temperature": 0.7, "max_tokens": 100}

    def test_binding_validates_model(self):
        """Test that binding validates model exists on creation."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.side_effect = Exception("Model 'invalid' not found")

            with pytest.raises(ModelNotFoundError) as exc_info:
                models.instance("invalid")

            assert "invalid" in str(exc_info.value)

    def test_call_binding(self, mock_model_response):
        """Test calling a binding with bound parameters."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()  # Model exists
            mock_registry.invoke_model.return_value = mock_model_response

            # Create binding
            binding = models.instance("gpt-4", temperature=0.7)

            # Call it
            result = binding("Hello world")

            assert result == "Test response"

            # Verify bound parameters were used
            mock_registry.invoke_model.assert_called_once_with(
                "gpt-4", "Hello world", temperature=0.7
            )

    def test_binding_response_helper(self, mock_model_response):
        """Binding.response should expose the structured response."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()  # Model exists
            mock_registry.invoke_model.return_value = mock_model_response

            binding = models.instance("gpt-4", temperature=0.7)

            response = binding.response("Hello world")

            assert isinstance(response, Response)
            assert response.text == "Test response"

    def test_binding_with_metadata(self, mock_model_response):
        """Binding.with_metadata should return a ModelInvocation wrapper."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()  # Model exists
            mock_registry.invoke_model.return_value = mock_model_response

            binding = models.instance("gpt-4", temperature=0.7)

            result = binding.with_metadata("Hello world")

            assert isinstance(result, ModelInvocation)
            assert result.text == "Test response"
            assert result.usage["total_tokens"] == 30
            assert result.model_id == "gpt-4"
            assert result.latency_ms == mock_model_response.latency_ms
            assert result.metadata["latency_ms"] == mock_model_response.latency_ms

    def test_override_parameters(self, mock_model_response):
        """Test overriding bound parameters on call."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()
            mock_registry.invoke_model.return_value = mock_model_response

            # Create binding with temperature=0.7
            binding = models.instance("gpt-4", temperature=0.7, max_tokens=50)

            # Call with override temperature=0.1
            result = binding("Hello", temperature=0.1)

            assert result == "Test response"

            # Override should take precedence
            mock_registry.invoke_model.assert_called_once_with(
                "gpt-4", "Hello", temperature=0.1, max_tokens=50
            )

    def test_multiple_calls_reuse_config(self, mock_model_response):
        """Test that bindings can be called multiple times."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()
            mock_registry.invoke_model.return_value = mock_model_response

            binding = models.instance("gpt-4", temperature=0.5)

            # First call
            result1 = binding("First prompt")
            assert result1 == "Test response"

            # Second call
            result2 = binding("Second prompt")
            assert result2 == "Test response"

            # Both calls should use same parameters
            assert mock_registry.invoke_model.call_count == 2
            calls = mock_registry.invoke_model.call_args_list
            assert calls[0][0] == ("gpt-4", "First prompt")
            assert calls[0][1] == {"temperature": 0.5}
            assert calls[1][0] == ("gpt-4", "Second prompt")
            assert calls[1][1] == {"temperature": 0.5}

    def test_binding_repr(self):
        """Test string representation of binding."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()

            binding = models.instance("gpt-4", temperature=0.7)

            repr_str = repr(binding)
            assert "ModelBinding" in repr_str
            assert "gpt-4" in repr_str
            assert "temperature" in repr_str
            assert "0.7" in repr_str

    def test_binding_with_no_parameters(self, mock_model_response):
        """Test creating binding without any preset parameters."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()
            mock_registry.invoke_model.return_value = mock_model_response

            # Binding with no parameters
            binding = models.instance("gpt-4")

            result = binding("Hello")

            # Should work with just model_id and prompt
            mock_registry.invoke_model.assert_called_once_with("gpt-4", "Hello")
            assert result == "Test response"

    def test_binding_error_propagation(self):
        """Test that errors propagate through bindings."""
        from ember._internal.exceptions import ProviderAPIError

        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.get_model.return_value = Mock()
            mock_registry.invoke_model.side_effect = ProviderAPIError("Rate limited")

            binding = models.instance("gpt-4", temperature=0.7)

            with pytest.raises(ProviderAPIError) as exc_info:
                binding("Hello")

            assert "Rate limited" in str(exc_info.value)
