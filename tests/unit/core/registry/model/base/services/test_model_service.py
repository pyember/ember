"""Unit tests for the ModelService.

Tests model retrieval and invocation using a dummy model provider.
"""

from typing import Any

import pytest

from ember.core.exceptions import ModelNotFoundError, ProviderAPIError
from ember.core.registry.model.base.registry.factory import ModelFactory
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.providers.base_provider import BaseProviderModel


# Define test providers directly in this module for test independence
class DummyServiceProvider(BaseProviderModel):
    """Test provider for service tests."""

    PROVIDER_NAME = "DummyService"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Process the request and return a response."""
        return ChatResponse(data=f"Echo: {request.prompt}")


class DummyAsyncProvider(BaseProviderModel):
    """Async test provider for service tests."""

    PROVIDER_NAME = "DummyAsyncService"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Process the request and return a response."""
        return ChatResponse(data=f"Async Echo: {request.prompt}")

    async def __call__(self, prompt: str, **kwargs: Any) -> ChatResponse:
        """Override to make this an async callable."""
        chat_request: ChatRequest = ChatRequest(prompt=prompt, **kwargs)
        return self.forward(request=chat_request)


class DummyErrorProvider(BaseProviderModel):
    """Provider that raises errors for testing."""

    PROVIDER_NAME = "DummyErrorService"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Always raise an error when called."""
        raise RuntimeError(f"Async error invoking model {self.model_info.id}")


# Register providers for test
@pytest.fixture(scope="function", autouse=True)
def register_test_providers(monkeypatch):
    """Register test providers for all tests in this module."""
    # Create a provider map for looking up classes
    test_providers = {
        "DummyService": DummyServiceProvider,
        "DummyAsyncService": DummyAsyncProvider,
        "DummyErrorService": DummyErrorProvider,
    }

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


# Note: We no longer need a custom DummyModel class here.
# We're using DummyServiceProvider from conftest.py instead.


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

    # We no longer need to patch these methods since we've properly
    # registered the DummyService provider in conftest.py
    registry.register_model(dummy_info)
    return registry


def test_get_model() -> None:
    """Test that ModelService.get_model retrieves the correct model."""
    # Create a minimal registry with direct mocking
    registry = ModelRegistry()

    # Create model info that will make the service query for our dummy model
    info = ModelInfo(
        id="dummy:service",
        name="Dummy Service",
        provider=ProviderInfo(name="DummyService"),
        cost=ModelCost(),
        rate_limit=RateLimit(),
        api_key="dummy",
    )
    registry.register_model(info)

    # Directly inject a model instance into registry's cache
    mock_model = DummyServiceProvider(model_info=info)
    registry._models["dummy:service"] = mock_model

    # Create service and test retrieval
    service = ModelService(registry=registry)
    model = service.get_model("dummy:service")

    # Verify it works
    response = model("hello")
    assert response.data == "Echo: hello"


def test_invoke_model() -> None:
    """Test that ModelService.invoke_model returns a ChatResponse with expected data."""
    # Create a minimal registry with direct mocking
    registry = ModelRegistry()

    # Create model info that will make the service query for our dummy model
    info = ModelInfo(
        id="dummy:service",
        name="Dummy Service",
        provider=ProviderInfo(name="DummyService"),
        cost=ModelCost(),
        rate_limit=RateLimit(),
        api_key="dummy",
    )
    registry.register_model(info)

    # Directly inject a model instance into registry's cache
    mock_model = DummyServiceProvider(model_info=info)
    registry._models["dummy:service"] = mock_model

    # Create service and test invoke
    service = ModelService(registry=registry)
    response = service.invoke_model(model_id="dummy:service", prompt="test prompt")

    # Verify it works
    assert "Echo: test prompt" in response.data


def test_get_model_invalid(dummy_registry: ModelRegistry) -> None:
    """Test that requesting an unregistered model raises a ModelNotFoundError."""
    service = ModelService(registry=dummy_registry)
    with pytest.raises(ModelNotFoundError):
        service.get_model("nonexistent:model")


# We'll add an async model provider in our conftest.py setup


# We'll add an error model provider in our conftest.py setup


# --- New Fixtures for Async Tests ---


@pytest.fixture
def dummy_async_registry() -> ModelRegistry:
    """Fixture to provide a registry with an async dummy model."""
    registry = ModelRegistry()
    async_info = create_dummy_model_info("dummy:async")
    # Update provider name to use our configured async provider
    async_info.provider.name = "DummyAsyncService"
    registry.register_model(async_info)
    return registry


@pytest.fixture
def dummy_error_registry() -> ModelRegistry:
    """Fixture that returns a ModelRegistry with a dummy error model registered."""
    registry = ModelRegistry()
    error_info = create_dummy_model_info("dummy:error")
    # Update provider name to use our configured error provider
    error_info.provider.name = "DummyErrorService"
    registry.register_model(error_info)
    return registry


# --- Async Test Cases ---


@pytest.mark.asyncio
async def test_invoke_model_async_sync() -> None:
    """Test async invocation for a synchronous dummy model using asyncio.to_thread."""
    # Create a minimal registry with direct mocking
    registry = ModelRegistry()

    # Create model info for our dummy model
    info = ModelInfo(
        id="dummy:service",
        name="Dummy Service",
        provider=ProviderInfo(name="DummyService"),
        cost=ModelCost(),
        rate_limit=RateLimit(),
        api_key="dummy",
    )
    registry.register_model(info)

    # Directly inject a model instance into registry's cache
    mock_model = DummyServiceProvider(model_info=info)
    registry._models["dummy:service"] = mock_model

    # Create service and test async invoke
    service = ModelService(registry=registry)
    response = await service.invoke_model_async(
        model_id="dummy:service", prompt="async test"
    )

    # Verify it works
    assert "Echo: async test" in response.data


@pytest.mark.asyncio
async def test_invoke_model_async_coroutine() -> None:
    """Test async invocation for a coroutine-based dummy model."""
    # Create a minimal registry with direct mocking
    registry = ModelRegistry()

    # Create model info for our dummy async model
    info = ModelInfo(
        id="dummy:async",
        name="Dummy Async Service",
        provider=ProviderInfo(name="DummyAsyncService"),
        cost=ModelCost(),
        rate_limit=RateLimit(),
        api_key="dummy",
    )
    registry.register_model(info)

    # Directly inject a model instance into registry's cache
    mock_model = DummyAsyncProvider(model_info=info)
    registry._models["dummy:async"] = mock_model

    # Create service and test async invoke
    service = ModelService(registry=registry)
    response = await service.invoke_model_async(
        model_id="dummy:async", prompt="async coroutine test"
    )

    # Verify it works
    assert "Async Echo: async coroutine test" in response.data


@pytest.mark.asyncio
async def test_invoke_model_async_error() -> None:
    """Test async invocation error handling when the model raises an exception."""
    # Create a minimal registry with direct mocking
    registry = ModelRegistry()

    # Create model info for our dummy error model
    info = ModelInfo(
        id="dummy:error",
        name="Dummy Error Service",
        provider=ProviderInfo(name="DummyErrorService"),
        cost=ModelCost(),
        rate_limit=RateLimit(),
        api_key="dummy",
    )
    registry.register_model(info)

    # Directly inject a model instance into registry's cache
    mock_model = DummyErrorProvider(model_info=info)
    registry._models["dummy:error"] = mock_model

    # Create service and test async invoke
    service = ModelService(registry=registry)
    with pytest.raises(ProviderAPIError) as exc_info:
        await service.invoke_model_async(
            model_id="dummy:error", prompt="test async error"
        )

    # Verify correct error message
    assert "Async error invoking model dummy:error" in str(exc_info.value)
