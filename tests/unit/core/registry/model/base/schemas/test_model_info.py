"""Unit tests for the ModelInfo schema.

Tests API key validation and the get_api_key method.
"""

import pytest

from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit


def create_model_info_with_key(model_id: str, api_key: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name="Test Model",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(name="TestProvider", default_api_key="default_key"),
        api_key=api_key,
    )


def create_model_info_without_key(model_id: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name="Test Model",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(name="TestProvider", default_api_key="default_key"),
        api_key=None,
    )


def test_get_api_key_explicit() -> None:
    """Test that get_api_key returns the explicitly provided API key."""
    model_info = create_model_info_with_key("test:model", "explicit_key")
    assert model_info.get_api_key() == "explicit_key"


def test_get_api_key_default() -> None:
    """Test that get_api_key returns the provider's default API key if explicit key is missing."""
    model_info = create_model_info_without_key("test:model")
    assert model_info.get_api_key() == "default_key"


def test_get_api_key_missing() -> None:
    """Test that missing both explicit and default API key raises a ValueError."""
    with pytest.raises(ValueError):
        ModelInfo(
            id="test:model",
            name="Test Model",
            cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
            rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
            provider=ProviderInfo(name="TestProvider", default_api_key=None),
            api_key=None,
        )
