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
from ember.core.registry.model.base.registry.factory import ModelFactory
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ProviderConfigError,
)


# Define test providers directly in this module for test independence
class MockProvider(BaseProviderModel):
    """Mock provider for model registry tests."""

    PROVIDER_NAME = "TestProvider"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Process the request and return a response."""
        return ChatResponse(data=request.prompt.upper())


# Register providers for test
@pytest.fixture(scope="function", autouse=True)
def register_test_providers(monkeypatch):
    """Register test providers for all tests in this module."""
    # Create a provider map for looking up classes
    test_providers = {"TestProvider": MockProvider}

    # Create a direct mock of the create_model_from_info method
    def mock_create_model_from_info(*, model_info):
        """Mock implementation that uses our test providers directly."""
        provider_name = model_info.provider.name
        if provider_name in test_providers:
            # Return an instance of our test provider
            provider_class = test_providers[provider_name]
            return provider_class(model_info=model_info)

        # For other providers, raise similar error as original
        available_providers = ", ".join(sorted(test_providers.keys()))
        raise ProviderConfigError(
            f"Unsupported provider '{provider_name}'. Available providers: {available_providers}"
        )

    # Apply the monkey patch
    monkeypatch.setattr(
        ModelFactory,
        "create_model_from_info",
        staticmethod(mock_create_model_from_info),
    )

    yield


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


def test_register_and_get_model() -> None:
    """Test that a model can be registered and then retrieved."""
    # Create new registry
    model_registry = ModelRegistry()

    # Create model info
    dummy_info = create_dummy_model_info("dummy:1")

    # Register the model
    model_registry.register_model(dummy_info)

    # Create and inject our mock provider directly
    mock_provider = MockProvider(model_info=dummy_info)
    model_registry._models["dummy:1"] = mock_provider

    # Retrieve and test
    retrieved = model_registry.get_model("dummy:1")
    response = retrieved("test")
    assert response.data == "TEST"


def test_register_duplicate_model() -> None:
    """Test that attempting to register a duplicate model raises a ValueError."""
    # Create new registry
    model_registry = ModelRegistry()

    # Create model info
    dummy_info = create_dummy_model_info("dummy:dup")

    # Register the model
    model_registry.register_model(dummy_info)

    # Try to register again and expect ValueError
    with pytest.raises(ValueError):
        model_registry.register_model(dummy_info)


def test_unregister_model() -> None:
    """Test that a registered model can be unregistered successfully."""
    # Create new registry
    model_registry = ModelRegistry()

    # Create model info
    dummy_info = create_dummy_model_info("dummy:unreg")

    # Register the model
    model_registry.register_model(dummy_info)

    # Create and inject our mock provider directly
    mock_provider = MockProvider(model_info=dummy_info)
    model_registry._models["dummy:unreg"] = mock_provider

    # Check it was registered
    assert "dummy:unreg" in model_registry.list_models()

    # Unregister the model
    model_registry.unregister_model("dummy:unreg")

    # Check it was unregistered
    assert "dummy:unreg" not in model_registry.list_models()

    # Should raise ModelNotFoundError when trying to access it
    with pytest.raises(ModelNotFoundError):
        model_registry.get_model("dummy:unreg")


def test_concurrent_registration():
    """Test thread-safe concurrent model registrations."""
    # Create new registry
    model_registry = ModelRegistry()

    # Monkeypatch the create_model_from_info method to avoid provider registry issues
    from ember.core.registry.model.base.registry.factory import ModelFactory

    original_method = ModelFactory.create_model_from_info
    # Just return the model_info itself as the model - no actual provider needed for this test
    ModelFactory.create_model_from_info = staticmethod(
        lambda *, model_info: MockProvider(model_info=model_info)
    )

    try:

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
    finally:
        # Restore original method
        ModelFactory.create_model_from_info = original_method
