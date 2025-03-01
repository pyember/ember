"""Unit tests for the ModelRegistry functionality.

Tests model registration, retrieval, listing, and unregistration.
"""

from typing import Any

import pytest
import threading

from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.exceptions import ModelNotFoundError
from ember.core.registry.model.base.schemas.model_info import (
    ModelInfo,
    ProviderInfo,
)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit


def create_dummy_model_info(model_id: str) -> ModelInfo:
    """Helper function to create a dummy ModelInfo instance for testing."""
    return ModelInfo(
        id=model_id,
        name=model_id,
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=ProviderInfo(name="TestProvider"),
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
    from ember.core.registry.model.base.registry.factory import ModelFactory

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
    from ember.core.registry.model.base.registry.factory import ModelFactory

    original_create = ModelFactory.create_model_from_info
    ModelFactory.create_model_from_info = staticmethod(lambda *, model_info: object())

    try:
        model_registry.register_model(dummy_info)
        with pytest.raises(ValueError):
            model_registry.register_model(dummy_info)
    finally:
        ModelFactory.create_model_from_info = original_create


def test_unregister_model(model_registry: ModelRegistry) -> None:
    """Test that a registered model can be unregistered successfully."""
    dummy_info = create_dummy_model_info("dummy:unreg")
    from ember.core.registry.model.base.registry.factory import ModelFactory

    ModelFactory.create_model_from_info = staticmethod(lambda *, model_info: object())

    model_registry.register_model(dummy_info)
    assert "dummy:unreg" in model_registry.list_models()
    model_registry.unregister_model("dummy:unreg")
    assert "dummy:unreg" not in model_registry.list_models()
    with pytest.raises(ModelNotFoundError):
        model_registry.get_model("dummy:unreg")


def test_concurrent_registration(model_registry):
    """Test thread-safe concurrent model registrations."""

    def register_model(model_id: str):
        info = create_dummy_model_info(model_id)
        model_registry.register_model(info)

    threads = [
        threading.Thread(target=register_model, args=(f"dummy:thread{i}",))
        for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(model_registry.list_models()) == 10
