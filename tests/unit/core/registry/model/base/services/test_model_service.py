"""Unit tests for the ModelService.

Tests model retrieval and invocation using a dummy model provider.
"""

from typing import Any

import pytest
import asyncio

from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.exceptions import ProviderAPIError, ModelNotFoundError


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
    from ember.core.registry.model.base.registry.factory import ModelFactory

    monkeypatch.setattr(
        ModelFactory,
        "create_model_from_info",
        lambda *, model_info: DummyModel(model_info),
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
    service = ModelService(registry=dummy_registry)
    with pytest.raises(ModelNotFoundError):
        service.get_model("nonexistent:model")


class DummyAsyncModel:
    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    async def __call__(self, prompt: str, **kwargs: Any) -> Any:
        class DummyAsyncResponse:
            data = f"Async Echo: {prompt}"
            usage = None

        return DummyAsyncResponse()


class DummyErrorModel:
    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        raise RuntimeError("Dummy error triggered")


# --- New Fixtures for Async Tests ---


@pytest.fixture
def dummy_async_registry(monkeypatch: pytest.MonkeyPatch) -> ModelRegistry:
    """Fixture to provide a registry with an async dummy model."""
    registry = ModelRegistry()
    async_info = create_dummy_model_info("dummy:async")
    from ember.core.registry.model.base.registry.factory import ModelFactory

    monkeypatch.setattr(
        ModelFactory,
        "create_model_from_info",
        lambda *, model_info: DummyAsyncModel(model_info),
    )
    registry.register_model(async_info)
    return registry


@pytest.fixture
def dummy_error_registry(monkeypatch: pytest.MonkeyPatch) -> ModelRegistry:
    """Fixture that returns a ModelRegistry with a dummy error model registered."""
    registry = ModelRegistry()
    error_info = create_dummy_model_info("dummy:error")
    from ember.core.registry.model.base.registry.factory import ModelFactory

    monkeypatch.setattr(
        ModelFactory,
        "create_model_from_info",
        lambda *, model_info: DummyErrorModel(model_info),
    )
    registry.register_model(error_info)
    return registry


# --- Async Test Cases ---


@pytest.mark.asyncio
async def test_invoke_model_async_sync(dummy_registry: ModelRegistry) -> None:
    """Test async invocation for a synchronous dummy model using asyncio.to_thread."""
    service = ModelService(registry=dummy_registry)
    response = await service.invoke_model_async(
        model_id="dummy:service", prompt="async test"
    )
    assert "Echo: async test" in response.data


@pytest.mark.asyncio
async def test_invoke_model_async_coroutine(
    dummy_async_registry: ModelRegistry,
) -> None:
    """Test async invocation for a coroutine-based dummy model."""
    service = ModelService(registry=dummy_async_registry)
    response = await service.invoke_model_async(
        model_id="dummy:async", prompt="async coroutine test"
    )
    assert "Async Echo: async coroutine test" in response.data


@pytest.mark.asyncio
async def test_invoke_model_async_error(dummy_error_registry: ModelRegistry) -> None:
    """Test async invocation error handling when the model raises an exception."""
    service = ModelService(registry=dummy_error_registry)
    with pytest.raises(ProviderAPIError) as exc_info:
        await service.invoke_model_async(
            model_id="dummy:error", prompt="test async error"
        )
    assert "Async error invoking model dummy:error" in str(exc_info.value)
