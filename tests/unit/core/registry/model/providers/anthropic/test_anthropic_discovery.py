"""Unit tests for the AnthropicDiscovery provider implementation.

These tests verify the behavior of the AnthropicDiscovery class, which is responsible
for discovering available models from the Anthropic API and standardizing their metadata.
"""

from typing import Dict, Any, List, Set
import os
import pytest

from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)


def test_anthropic_discovery_fetch_models() -> None:
    """Test that AnthropicDiscovery returns the expected simulated model info.
    
    This test verifies that:
    1. The expected model IDs are present in the returned data
    2. Each model has the correct model_id and model_name format
    3. The model names match the naming convention used in the implementation
    """
    discovery = AnthropicDiscovery()
    models: Dict[str, Dict[str, Any]] = discovery.fetch_models()
    
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
    assert entry.get("model_name").startswith("claude-3.5-sonnet")
    assert "api_data" in entry


def test_anthropic_fallback_models() -> None:
    """Test that fallback models are provided when discovery fails."""
    discovery = AnthropicDiscovery()
    
    # Override environment to ensure we get fallback models
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    
    models = discovery.fetch_models()
    
    # Verify we still get the essential models
    assert len(models) >= 4
    assert "anthropic:claude-3-opus" in models
    assert "anthropic:claude-3-sonnet" in models
    
    monkeypatch.undo()


def test_anthropic_discovery_model_id_generation() -> None:
    """Test the model ID generation logic for Anthropic models."""
    discovery = AnthropicDiscovery()
    
    # Access private method for testing
    model_id = discovery._generate_model_id(raw_model_id="claude-3-opus-20240229")
    assert model_id == "anthropic:claude-3-opus"
    
    model_id = discovery._generate_model_id(raw_model_id="claude-3.5-sonnet-20240620")
    assert model_id == "anthropic:claude-3.5-sonnet"


def test_anthropic_discovery_fetch_models_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test error handling when fetch_models encounters an exception."""
    def mock_fetch_models_error() -> Dict[str, Any]:
        """Mock function that raises an exception."""
        raise Exception("Discovery error")

    discovery = AnthropicDiscovery()
    monkeypatch.setattr(discovery, "_get_client", mock_fetch_models_error)
    
    # Should return fallback models instead of raising an exception
    models = discovery.fetch_models()
    assert len(models) > 0
    assert "anthropic:claude-3-opus" in models
