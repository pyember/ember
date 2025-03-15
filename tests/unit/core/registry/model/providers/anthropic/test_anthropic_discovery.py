"""Unit tests for the AnthropicDiscovery provider implementation.

These tests verify the behavior of the AnthropicDiscovery class, which is responsible
for discovering available models from the Anthropic API and standardizing their metadata.
"""

from typing import Dict, Any, List, Set
import os
import pytest
import sys
import types

# Create mock module for anthropic
mock_anthropic = types.ModuleType("anthropic")

# Create mock Anthropic class
class MockAnthropic:
    def __init__(self, **kwargs):
        pass

# Add to the mock module
mock_anthropic.Anthropic = MockAnthropic

# Replace the real module with our mock
sys.modules["anthropic"] = mock_anthropic

# Import after mocking
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)

# Create a subclass to avoid app_context access
class TestAnthropicDiscovery(AnthropicDiscovery):
    """Test-specific subclass that avoids app_context."""
    
    def _get_client(self):
        """Override to avoid app_context access."""
        if self._client is not None:
            return self._client
            
        if not self._api_key:
            return None
            
        try:
            self._client = mock_anthropic.Anthropic(api_key=self._api_key)
            return self._client
        except Exception:
            return None


@pytest.fixture
def discovery_instance():
    """Return a preconfigured discovery instance."""
    discovery = TestAnthropicDiscovery()
    discovery.configure(api_key="test-key")
    return discovery


def test_anthropic_discovery_fetch_models(discovery_instance) -> None:
    """Test that AnthropicDiscovery returns the expected simulated model info."""
    models: Dict[str, Dict[str, Any]] = discovery_instance.fetch_models()
    
    # Verify essential models are present
    expected_models = {
        "anthropic:claude-3-opus",
        "anthropic:claude-3-sonnet", 
        "anthropic:claude-3-haiku",
        "anthropic:claude-3.5-sonnet"
    }
    actual_models = set(models.keys())
    assert expected_models.issubset(actual_models), (
        f"Missing expected models. Expected at least: {expected_models}, "
        f"got: {actual_models}"
    )
    
    # Verify specific model structure
    model_id = "anthropic:claude-3.5-sonnet"
    entry = models[model_id]
    assert entry.get("model_id") == model_id
    assert entry.get("model_name") is not None
    assert "api_data" in entry


def test_anthropic_fallback_models() -> None:
    """Test that fallback models are provided when discovery fails."""
    # Create a fresh instance explicitly setting empty API key
    discovery = TestAnthropicDiscovery()
    discovery.configure(api_key="")  # Empty API key
    
    models = discovery.fetch_models()
    
    # Verify we still get the essential models through fallback
    assert len(models) >= 4
    assert "anthropic:claude-3-opus" in models
    assert "anthropic:claude-3-sonnet" in models


def test_anthropic_discovery_model_id_generation(discovery_instance) -> None:
    """Test the model ID generation logic for Anthropic models."""
    # Access private method for testing
    model_id = discovery_instance._generate_model_id(raw_model_id="claude-3-opus-20240229")
    assert model_id == "anthropic:claude-3-opus"
    
    model_id = discovery_instance._generate_model_id(raw_model_id="claude-3.5-sonnet-20240620")
    assert model_id == "anthropic:claude-3.5-sonnet"


def test_anthropic_discovery_fetch_models_error(
    discovery_instance, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test error handling when fetch_models encounters an exception."""
    def mock_fetch_models_error() -> Dict[str, Any]:
        """Mock function that raises an exception."""
        raise Exception("Discovery error")

    monkeypatch.setattr(discovery_instance, "_get_client", mock_fetch_models_error)
    
    # Should return fallback models instead of raising an exception
    models = discovery_instance.fetch_models()
    assert len(models) > 0
    assert "anthropic:claude-3-opus" in models
