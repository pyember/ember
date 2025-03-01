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


def test_deepmind_discovery_fetch_models() -> None:
    """Test that DeepmindDiscovery.fetch_models returns a correctly prefixed model dict."""
    discovery = DeepmindDiscovery()
    models: Dict[str, Dict[str, Any]] = discovery.fetch_models()
    key = "google:gemini-1.5-pro"
    assert key in models
    entry = models[key]
    assert entry.get("model_id") == key
    assert entry.get("model_name") == "gemini-1.5-pro"


def test_deepmind_discovery_fetch_models_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that if genai.list_models throws an error, fetch_models handles it gracefully."""
    monkeypatch.setattr(
        genai, "list_models", lambda: (_ for _ in ()).throw(Exception("API error"))
    )
    discovery = DeepmindDiscovery()
    models = discovery.fetch_models()
    assert models == {}
