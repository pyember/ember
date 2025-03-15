"""Unit tests for the OpenAIDiscovery provider implementation.
This test mocks openai.Model.list() to simulate API responses.
"""

import pytest
from typing import Any, Dict

import openai

from ember.core.registry.model.providers.openai.openai_discovery import (
    OpenAIDiscovery,
)


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch openai.Model.list to return a dummy response."""

    def mock_model_list() -> Dict[str, Any]:
        return {
            "data": [
                {"id": "gpt-4o", "name": "gpt-4o"},
                {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
            ]
        }

    monkeypatch.setattr(openai.Model, "list", mock_model_list)
    # Mock app_context to avoid circular dependency
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def discovery_instance():
    """Return a preconfigured discovery instance to avoid app_context access."""
    discovery = OpenAIDiscovery()
    discovery.configure(api_key="test-key")
    return discovery


def test_openai_discovery_fetch_models(discovery_instance) -> None:
    """Test that OpenAIDiscovery fetches and standardizes model metadata."""
    models = discovery_instance.fetch_models()
    for expected_key in ["openai:gpt-4o", "openai:gpt-4o-mini"]:
        assert expected_key in models
        assert models[expected_key]["model_id"] == expected_key
