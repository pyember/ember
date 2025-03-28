"""
Unit tests for the usage_example script.
"""

import logging
from unittest.mock import MagicMock

import pytest


class DummyResponse:
    """Simple response object with data and usage attributes."""

    def __init__(self, text: str):
        self.data = text
        self.usage = None


@pytest.fixture
def mock_model_service():
    """Create a mock model service for testing."""
    mock_service = MagicMock()

    # Setup invoke_model method
    def side_effect(model_id, prompt, **kwargs):
        model_name = getattr(model_id, "value", model_id)
        return DummyResponse(f"Response from {model_name}: {prompt}")

    mock_service.invoke_model.side_effect = side_effect

    # Setup get_model method
    mock_model = MagicMock()
    mock_model.side_effect = lambda prompt, **kwargs: DummyResponse(
        f"Response from model: {prompt}"
    )
    mock_service.get_model.return_value = mock_model

    return mock_service


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch, mock_model_service):
    """Mock all dependencies needed by the usage_example module."""
    # Mock the get_model_service function
    monkeypatch.setattr(
        "ember.core.registry.model.examples.usage_example.get_model_service",
        lambda *args, **kwargs: mock_model_service,
    )

    # Suppress logging
    monkeypatch.setattr(
        "ember.core.registry.model.examples.usage_example.logging.basicConfig",
        lambda **kwargs: None,
    )


def test_usage_example_output(capsys):
    """Test that the usage_example main function produces expected output."""
    # Suppress all logging
    logging.getLogger().handlers = [logging.NullHandler()]

    # Import here to avoid module-level import issues
    from ember.core.registry.model.examples.usage_example import main

    # Run the main function
    main()

    # Check output
    captured = capsys.readouterr().out
    assert "Response using string ID:" in captured
    assert "Response using Enum:" in captured
