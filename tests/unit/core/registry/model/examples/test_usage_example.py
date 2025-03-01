"""
Unit tests for the usage_example script.
Uses a dummy init() to return a dummy ModelService so that the example main() produces expected output.
"""

import io
import logging
from typing import Any

import pytest

# Assume the usage_example script's main() is importable:
from ember.core.registry.model.examples.usage_example import main


class DummyModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        class DummyResponse:
            data = f"Response from {self.model_id}: {prompt}"
            usage = None

        return DummyResponse()


class DummyModelService:
    def __init__(self) -> None:
        self.models = {"openai:gpt-4o": DummyModel("openai:gpt-4o")}

    def __call__(self, model_id: str, prompt: str, **kwargs: Any) -> Any:
        return self.models.get(model_id, DummyModel(model_id))(prompt)

    def get_model(self, model_id: str) -> Any:
        return self.models.get(model_id, DummyModel(model_id))


class DummyInitService:
    """A dummy init() that accepts usage_tracking and returns a DummyModelService."""

    def __call__(
        self, usage_tracking: bool = True, *args, **kwargs
    ) -> DummyModelService:
        return DummyModelService()


@pytest.fixture(autouse=True)
def patch_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch usage_example.py so that 'init(...)' is replaced by DummyInitService(),
    which accepts usage_tracking without raising a TypeError.
    """
    from ember.core.registry.model.examples import usage_example

    monkeypatch.setattr(usage_example, "init", DummyInitService())


def test_usage_example_output(capsys: pytest.CaptureFixture) -> None:
    """Run the usage_example main() and capture its output for expected strings."""
    logging.getLogger().handlers = [logging.NullHandler()]
    main()
    captured = capsys.readouterr().out
    assert "Response using string ID:" in captured or "Response using Enum:" in captured
