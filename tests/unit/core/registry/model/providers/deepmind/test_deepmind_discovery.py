"""Unit tests for the DeepmindDiscovery provider implementation.
This test mocks google.generativeai.list_models to return a dummy model.
"""

from typing import Any, Dict

import google.generativeai as genai
import pytest

from ember.core.registry.model.providers.deepmind.deepmind_discovery import (
    DeepmindDiscovery,
)


class DummyModel:
    """Dummy model class for testing."""

    name = "gemini-1.5-pro"
    supported_generation_methods = ["generateContent"]

    def __init__(self, name=None, supported_methods=None):
        if name:
            self.name = name
        if supported_methods is not None:  # Check for None, not falsiness
            self.supported_generation_methods = supported_methods


@pytest.fixture(autouse=True)
def patch_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch genai.list_models to return a list with dummy models."""

    def mock_list_models():
        """Return a list of dummy models for testing."""
        return [
            DummyModel("gemini-1.5-pro"),
            DummyModel("gemini-1.5-flash"),
            DummyModel("gemini-2.0-pro"),  # Add new models from the enum
            DummyModel("gemini-2.0-flash"),  # Add new models from the enum
            # This one should be filtered out due to missing generateContent
            DummyModel("other-model", []),
        ]

    monkeypatch.setattr(genai, "list_models", mock_list_models)
    # Mock environment to avoid app_context dependency
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # Mock genai.configure to prevent actual API calls
    monkeypatch.setattr(genai, "configure", lambda **kwargs: None)


@pytest.fixture
def discovery_instance():
    """Return a preconfigured discovery instance."""
    discovery = DeepmindDiscovery()
    discovery.configure(api_key="test-key")
    # Mark as initialized to avoid app_context API key lookup
    discovery._initialized = True
    return discovery


def test_deepmind_discovery_fetch_models(discovery_instance) -> None:
    """Test that DeepmindDiscovery.fetch_models returns a correctly prefixed model dict."""
    models: Dict[str, Dict[str, Any]] = discovery_instance.fetch_models()

    # Should include models with generateContent support
    assert "deepmind:gemini-1.5-pro" in models
    assert "deepmind:gemini-1.5-flash" in models
    assert "deepmind:gemini-2.0-pro" in models
    assert "deepmind:gemini-2.0-flash" in models

    # Should filter out models without generateContent support
    assert "deepmind:other-model" not in models

    # Verify model structure
    entry = models["deepmind:gemini-1.5-pro"]
    assert entry.get("id") == "deepmind:gemini-1.5-pro"
    assert entry.get("name") == "gemini-1.5-pro"


def test_deepmind_discovery_fetch_models_error(
    discovery_instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that if genai.list_models throws an error, fetch_models handles it gracefully."""

    def mock_list_models_error():
        """Mock function that throws an exception."""
        raise Exception("API error")

    monkeypatch.setattr(genai, "list_models", mock_list_models_error)

    # Check error handling behavior - should return empty dict
    models = discovery_instance.fetch_models()
    assert isinstance(models, dict)
    assert len(models) == 0  # No fallbacks anymore
