"""Unit tests for the DeepmindDiscovery provider implementation.
This test mocks google.generativeai.list_models to return a dummy model.
"""

import pytest
from typing import Any, Dict

import google.generativeai as genai

from ember.core.registry.model.providers.deepmind.deepmind_discovery import (
    DeepmindDiscovery,
)


class DummyModel:
    name = "gemini-1.5-pro"


@pytest.fixture(autouse=True)
def patch_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch genai.list_models to return a list with a dummy model."""
    monkeypatch.setattr(genai, "list_models", lambda: [DummyModel()])
    # Mock environment to avoid app_context dependency
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    # Mock genai.configure to prevent actual API calls
    monkeypatch.setattr(genai, "configure", lambda api_key: None)


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
    key = "google:gemini-1.5-pro"
    assert key in models
    entry = models[key]
    assert entry.get("model_id") == key
    assert entry.get("model_name") == "gemini-1.5-pro"


def test_deepmind_discovery_fetch_models_error(
    discovery_instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that if genai.list_models throws an error, fetch_models handles it gracefully."""
    monkeypatch.setattr(
        genai, "list_models", lambda: (_ for _ in ()).throw(Exception("API error"))
    )

    # Check fallback behavior
    models = discovery_instance.fetch_models()
    assert len(models) > 0
    # It should return fallback models
    assert "google:gemini-1.5-pro" in models
    assert "google:gemini-pro" in models
