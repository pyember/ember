"""Unit tests for the AnthropicDiscovery provider implementation.
Since AnthropicDiscovery.fetch_models returns simulated data, we just verify its content.
"""

import pytest
from typing import Dict, Any

from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)


def test_anthropic_discovery_fetch_models() -> None:
    """Test that AnthropicDiscovery returns the simulated model info."""
    discovery = AnthropicDiscovery()
    models: Dict[str, Dict[str, Any]] = discovery.fetch_models()
    expected_key = "anthropic:claude-3.5-sonnet"
    assert expected_key in models
    entry = models[expected_key]
    assert entry.get("model_id") == expected_key
    assert entry.get("model_name") == "claude-3.5-sonnet-latest"


def test_anthropic_discovery_fetch_models_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that if fetch_models raises an error, it is handled gracefully."""

    def mock_fetch_models_error() -> Dict[str, Any]:
        raise Exception("Discovery error")

    monkeypatch.setattr(AnthropicDiscovery, "fetch_models", mock_fetch_models_error)
    discovery = AnthropicDiscovery()
    with pytest.raises(Exception):
        discovery.fetch_models()
