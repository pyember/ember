#!/usr/bin/env python3
"""Unit tests for the ModelRegistry functionality.

Tests model registration, retrieval, listing, and unregistration.
"""

from typing import Any, Dict

import pytest

from src.ember.core.registry.model.registry.model_registry import ModelRegistry
from src.ember.core.registry.model.schemas.model_info import ModelInfo
from src.ember.core.registry.model.schemas.provider_info import ProviderInfo
from src.ember.core.registry.model.schemas.cost import ModelCost, RateLimit


def create_dummy_model_info(model_id: str = "dummy:1") -> ModelInfo:
    """Helper function to create a dummy ModelInfo instance for testing."""
    return ModelInfo(
        id=model_id,
        name="Dummy Model",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(name="Dummy", default_api_key="dummy_key"),
        api_key="dummy_key",
    )


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Fixture that returns a fresh ModelRegistry instance."""
    return ModelRegistry()


def test_register_and_get_model(model_registry: ModelRegistry) -> None:
    """Test that a model can be registered and then retrieved."""
    dummy_info = create_dummy_model_info("dummy:1")

    # Define a dummy provider that echoes the prompt in uppercase.
    class DummyProvider:
        def __init__(self, model_info: ModelInfo) -> None:
            self.model_info = model_info

        def __call__(self, prompt: str, **kwargs: Any) -> str:
            return prompt.upper()

    # Monkey-patch ModelFactory.create_model_from_info to return our dummy provider.
    from src.ember.core.registry.model.registry.factory import ModelFactory

    original_create = ModelFactory.create_model_from_info
    ModelFactory.create_model_from_info = staticmethod(
        lambda *, model_info: DummyProvider(model_info)
    )

    try:
        model_registry.register_model(dummy_info)
        retrieved = model_registry.get_model("dummy:1")
        assert retrieved("test") == "TEST"
    finally:
        ModelFactory.create_model_from_info = original_create


def test_register_duplicate_model(model_registry: ModelRegistry) -> None:
    """Test that attempting to register a duplicate model raises a ValueError."""
    dummy_info = create_dummy_model_info("dummy:dup")
    from src.ember.core.registry.model.registry.factory import ModelFactory

    original_create = ModelFactory.create_model_from_info
    ModelFactory.create_model_from_info = staticmethod(lambda *, model_info: object())

    try:
        model_registry.register_model(dummy_info)
        with pytest.raises(ValueError):
            model_registry.register_model(dummy_info)
    finally:
        ModelFactory.create_model_from_info = original_create


def test_get_model_not_found(model_registry: ModelRegistry) -> None:
    """Test that requesting an unregistered model raises an exception."""
    with pytest.raises(Exception):
        model_registry.get_model("nonexistent:model")


def test_unregister_model(model_registry: ModelRegistry) -> None:
    """Test that a registered model can be unregistered successfully."""
    dummy_info = create_dummy_model_info("dummy:unreg")
    from src.ember.core.registry.model.registry.factory import ModelFactory

    ModelFactory.create_model_from_info = staticmethod(lambda *, model_info: object())

    model_registry.register_model(dummy_info)
    assert "dummy:unreg" in model_registry.list_models()
    model_registry.unregister_model("dummy:unreg")
    assert "dummy:unreg" not in model_registry.list_models()
