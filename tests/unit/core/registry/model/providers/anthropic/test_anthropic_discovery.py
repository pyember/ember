"""Unit tests for the AnthropicDiscovery provider implementation.

These tests verify the behavior of the AnthropicDiscovery class, which is responsible
for discovering available models from the Anthropic API and standardizing their metadata.
"""

import sys
import types
from typing import Any, Dict

import pytest

# Create mock module for anthropic
mock_anthropic = types.ModuleType("anthropic")


# Create mock Anthropic class with model list support
class MockAnthropic:
    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")

    @property
    def models(self):
        return self.Models()

    class Models:
        def list(self):
            return MockModelResponse()


class MockModel:
    def __init__(self, id):
        self.id = id


class MockModelResponse:
    @property
    def data(self):
        return [
            MockModel("claude-3-opus-20240229"),
            MockModel("claude-3-haiku-20240307"),
            MockModel("claude-3-5-sonnet-20240620"),
            MockModel("claude-3.7-sonnet-20250219"),
        ]


# Add to the mock module
mock_anthropic.Anthropic = MockAnthropic
mock_anthropic.APIError = Exception
mock_anthropic.RateLimitError = Exception
mock_anthropic.APIStatusError = Exception
mock_anthropic.APIConnectionError = Exception

# Replace the real module with our mock
sys.modules["anthropic"] = mock_anthropic

# Import after mocking
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)


@pytest.fixture
def discovery_instance():
    """Return a discovery instance initialized with a test API key."""
    # Pass the API key to the constructor
    return AnthropicDiscovery(api_key="test-key")


def test_anthropic_discovery_fetch_models(discovery_instance, monkeypatch) -> None:
    """Test that AnthropicDiscovery returns the expected model info."""
    # Mock the requests.get method to return a controlled response
    from unittest.mock import MagicMock

    import requests

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"id": "claude-3-opus-20240229", "object": "model"},
            {"id": "claude-3-haiku-20240307", "object": "model"},
            {"id": "claude-3-5-sonnet-20240620", "object": "model"},
            {"id": "claude-3.7-sonnet-20250219", "object": "model"},
        ]
    }
    mock_response.raise_for_status = lambda: None

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)

    # Test model fetching with our mock
    models: Dict[str, Dict[str, Any]] = discovery_instance.fetch_models()

    # For this test, we're using our mocked response, so we should get specific models
    expected_models = {
        "anthropic:claude-3-opus",  # Base model extracted from claude-3-opus-20240229
        "anthropic:claude-3-haiku",  # Base model extracted from claude-3-haiku-20240307
        "anthropic:claude-3-5-sonnet",  # Base model extracted from claude-3-5-sonnet-20240620
        "anthropic:claude-3.7-sonnet",  # Base model extracted from claude-3.7-sonnet-20250219
    }

    actual_models = set(models.keys())
    for model in expected_models:
        assert model in actual_models, f"Missing expected model {model}"

    # Verify structure of one model
    model_id = "anthropic:claude-3-5-sonnet"
    entry = models[model_id]
    assert entry.get("id") == model_id
    assert entry.get("name") is not None
    assert "api_data" in entry


def test_anthropic_fallback_models() -> None:
    """Test that fallback models are provided when API key is missing."""
    # Save the original environment variable value to restore later
    import os

    original_api_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        # Clear the environment variable
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        # Import after clearing environment variable
        from ember.core.registry.model.providers.base_discovery import (
            ModelDiscoveryError,
        )

        # Now the constructor should raise ModelDiscoveryError
        with pytest.raises(ModelDiscoveryError):
            AnthropicDiscovery()
    finally:
        # Restore the original environment variable if it existed
        if original_api_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = original_api_key


def test_anthropic_discovery_model_id_generation(discovery_instance) -> None:
    """Test the model ID generation logic for Anthropic models."""
    # Test the model ID generation
    model_id = discovery_instance._generate_model_id("claude-3-opus-20240229")
    assert model_id == "anthropic:claude-3-opus-20240229"


def test_extract_base_model_id(discovery_instance) -> None:
    """Test the extraction of base model IDs from dated model identifiers."""
    # Test that model variants are properly preserved
    assert (
        discovery_instance._extract_base_model_id("claude-3-opus-20240229")
        == "claude-3-opus"
    )
    assert (
        discovery_instance._extract_base_model_id("claude-3-haiku-20240307")
        == "claude-3-haiku"
    )
    assert (
        discovery_instance._extract_base_model_id("claude-3-5-sonnet-20241022")
        == "claude-3-5-sonnet"
    )
    assert (
        discovery_instance._extract_base_model_id("claude-3.7-sonnet-20250219")
        == "claude-3.7-sonnet"
    )

    # Test undated model IDs remain unchanged
    assert (
        discovery_instance._extract_base_model_id("claude-3-5-sonnet")
        == "claude-3-5-sonnet"
    )
    assert (
        discovery_instance._extract_base_model_id("claude-3.7-sonnet")
        == "claude-3.7-sonnet"
    )

    # Test base model names
    assert discovery_instance._extract_base_model_id("claude-3") == "claude-3"
    assert discovery_instance._extract_base_model_id("claude-3.5") == "claude-3.5"


def test_anthropic_discovery_fetch_models_error(
    discovery_instance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test error handling when fetch_models encounters an exception."""

    # Instead of patching models.list, patching the Anthropic client itself
    # to throw an exception when any method is accessed
    def mock_client_that_raises(*args, **kwargs):
        raise Exception("API error")

    # Replacing the entire client with our mock
    monkeypatch.setattr(discovery_instance, "client", mock_client_that_raises)

    # Should return an empty dictionary with no fallback models
    models = discovery_instance.fetch_models()
    assert isinstance(models, dict)
    assert len(models) == 0


def test_anthropic_discovery_timeout_handling(
    discovery_instance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the improved timeout handling in the Anthropic provider."""

    # Mock requests.get to raise a timeout exception
    def request_that_raises_timeout(*args, **kwargs):
        import requests

        raise requests.exceptions.Timeout("Connection timed out")

    import requests

    monkeypatch.setattr(requests, "get", request_that_raises_timeout)

    # Test that we handle the timeout properly and return empty dict
    result = discovery_instance.fetch_models()

    # Verify that no models are returned
    assert isinstance(result, dict)
    assert len(result) == 0


def test_anthropic_discovery_response_format_handling(
    discovery_instance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test handling of different response formats from the Anthropic API."""
    from unittest.mock import MagicMock

    # Case 1: Test with response containing dict with data key
    dict_response = MagicMock()
    dict_response.json.return_value = {
        "data": [
            {
                "id": "claude-3-opus-20240229",
                "object": "model",
                "display_name": "Claude 3 Opus",
            },
            {
                "id": "claude-3-sonnet-20240229",
                "object": "model",
                "display_name": "Claude 3 Sonnet",
            },
        ]
    }
    dict_response.raise_for_status = lambda: None

    # Case 2: Test with response containing a direct list
    list_response = MagicMock()
    list_response.json.return_value = [
        {
            "id": "claude-3-haiku-20240307",
            "object": "model",
            "display_name": "Claude 3 Haiku",
        },
    ]
    list_response.raise_for_status = lambda: None

    # Mock requests.get to return our test responses in sequence
    response_sequence = [dict_response, list_response]
    response_index = 0

    def mock_requests_get(*args, **kwargs):
        nonlocal response_index
        response = response_sequence[response_index % len(response_sequence)]
        response_index += 1
        return response

    import requests

    monkeypatch.setattr(requests, "get", mock_requests_get)

    # Test with dictionary response format
    dict_result = discovery_instance.fetch_models()
    assert "anthropic:claude-3-opus" in dict_result
    assert "anthropic:claude-3-sonnet" in dict_result

    # Test with list response format
    list_result = discovery_instance.fetch_models()
    assert "anthropic:claude-3-haiku" in list_result
