"""Unit tests for the ModelFactory.

Tests instantiation of provider models and error conditions.
"""

import pytest
from typing import Any, Dict
from unittest.mock import patch

from ember.core.registry.model.base.registry.factory import ModelFactory
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ProviderConfigError,
)
from ember.core.registry.model.providers.base_provider import BaseProviderModel


# Dummy provider class for testing
class DummyProviderModel(BaseProviderModel):
    PROVIDER_NAME = "DummyProvider"

    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return f"Echo: {prompt}"

    def create_client(self) -> None:
        return None

    def forward(self, request):
        return None


# Dummy discover function that returns our dummy provider.
def dummy_discover_providers(*, package_path: str) -> Dict[str, type]:
    return {"DummyProvider": DummyProviderModel}


@pytest.fixture(autouse=True)
def patch_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the factory for testing purposes.
    
    This function directly patches the ModelFactory implementation since
    monkeypatching the import path has issues with duplicate modules.
    """
    # Import factory module directly
    from ember.core.registry.model.base.registry import factory
    
    # Apply patch directly to the module
    monkeypatch.setattr(factory, "discover_providers_in_package", dummy_discover_providers)
    
    # Also reset the cached providers
    from ember.core.registry.model.base.registry.factory import ModelFactory
    ModelFactory._provider_cache = None


def create_dummy_model_info(model_id: str = "openai:gpt-4o") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name="DummyFactoryModel",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(name="DummyProvider", default_api_key="dummy_key"),
        api_key="dummy_key",
    )


@patch.object(
    ModelFactory, "_get_providers", return_value={"DummyProvider": DummyProviderModel}
)
def test_create_model_from_info_success(mock_get_providers) -> None:
    """Test that ModelFactory.create_model_from_info successfully creates a DummyProviderModel."""
    dummy_info = create_dummy_model_info("openai:gpt-4o")
    model_instance = ModelFactory.create_model_from_info(model_info=dummy_info)
    assert isinstance(model_instance, DummyProviderModel)
    assert model_instance.model_info.id == "openai:gpt-4o"


def test_create_model_from_info_invalid() -> None:
    """Test that an invalid model id causes ProviderConfigError."""
    dummy_info = create_dummy_model_info("invalid:model")

    # Use patch to avoid import path issues
    from ember.core.registry.model.base.registry import factory
    from ember.core.registry.model.config import model_enum
    
    # Keep original function for restoration after test
    original_parse_func = model_enum.parse_model_str
    
    try:
        # Directly modify the function
        def mock_parse_model_str(model_str: str) -> str:
            raise ValueError("Invalid model ID format")
            
        # Apply the patch directly to the imported module
        model_enum.parse_model_str = mock_parse_model_str
        
        # The factory imports parse_model_str directly, so we need to patch that reference too
        factory.parse_model_str = mock_parse_model_str
        
        # Now test the factory function - should raise ProviderConfigError
        with pytest.raises(ProviderConfigError):
            ModelFactory.create_model_from_info(model_info=dummy_info)
    finally:
        # Restore the original function
        model_enum.parse_model_str = original_parse_func
        factory.parse_model_str = original_parse_func
