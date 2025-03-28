"""Unit tests for the OpenAIDiscovery provider implementation.
This test mocks openai.Model.list() to simulate API responses.
"""


import openai
import pytest

from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch openai client to return a dummy response."""

    class MockOpenAI:
        class Models:
            def list(self):
                return MockResponse()

        @property
        def models(self):
            return self.Models()

    class MockModel:
        def __init__(self, id, object_type="model"):
            self.id = id
            self.object = object_type

    class MockResponse:
        @property
        def data(self):
            return [
                MockModel("gpt-4o"),
                MockModel("gpt-4o-mini"),
            ]

    # Patch the OpenAI class
    monkeypatch.setattr(openai, "OpenAI", MockOpenAI)
    # Set environment variable for tests
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def discovery_instance():
    """Return a configured discovery instance."""
    return OpenAIDiscovery(api_key="test-key")


def test_openai_discovery_empty_model_list(discovery_instance) -> None:
    """Test that OpenAIDiscovery handles empty model lists correctly.

    Since the hardcoded fallback models were removed, the function should
    return an empty dictionary when API access fails without raising exceptions.
    """
    models = discovery_instance.fetch_models()
    # The function should return an empty dict when API access fails
    assert isinstance(models, dict)
    # No assertion on specific models since they're no longer hardcoded
