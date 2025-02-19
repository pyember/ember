#!/usr/bin/env python3
"""Unit tests for the ModelService.

Tests model retrieval and invocation using a dummy model provider.
"""

from typing import Any

import pytest

from src.ember.core.registry.model.services.model_service import ModelService
from src.ember.core.registry.model.registry.model_registry import ModelRegistry
from src.ember.core.registry.model.schemas.model_info import ModelInfo
from src.ember.core.registry.model.schemas.provider_info import ProviderInfo
from src.ember.core.registry.model.schemas.cost import ModelCost, RateLimit


# Dummy model that echoes the prompt.
class DummyModel:
    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        class DummyResponse:
            data = f"Echo: {prompt}"
            usage = None
        return DummyResponse()


def create_dummy_model_info(model_id: str = "dummy:service") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name="Dummy Service Model",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(name="DummyService", default_api_key="dummy_key"),
        api_key="dummy_key",
    )


@pytest.fixture
def dummy_registry(monkeypatch: pytest.MonkeyPatch) -> ModelRegistry:
    """Fixture that returns a ModelRegistry with a dummy model registered."""
    registry = ModelRegistry()
    dummy_info = create_dummy_model_info("dummy:service")
    from src.ember.core.registry.model.registry.factory import ModelFactory
    monkeypatch.setattr(
        ModelFactory, "create_model_from_info", lambda *, model_info: DummyModel(model_info)
    )
    registry.register_model(dummy_info)
    return registry


def test_get_model(dummy_registry: ModelRegistry) -> None:
    """Test that ModelService.get_model retrieves the correct model."""
    service = ModelService(registry=dummy_registry)
    model = service.get_model("dummy:service")
    response = model("hello")
    assert response.data == "Echo: hello"


def test_invoke_model(dummy_registry: ModelRegistry) -> None:
    """Test that ModelService.invoke_model returns a ChatResponse with expected data."""
    service = ModelService(registry=dummy_registry)
    response = service.invoke_model(model_id="dummy:service", prompt="test prompt")
    assert "Echo: test prompt" in response.data


def test_get_model_invalid(dummy_registry: ModelRegistry) -> None:
    """Test that requesting an unregistered model raises a ModelNotFoundError."""
    from src.ember.core.exceptions import ModelNotFoundError
    service = ModelService(registry=dummy_registry)
    with pytest.raises(ModelNotFoundError):
        service.get_model("nonexistent:model")