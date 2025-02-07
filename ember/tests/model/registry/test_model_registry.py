import pytest

from src.ember.registry.model.registry.model_registry import ModelRegistry
from src.ember.registry.model.schemas.model_info import ModelInfo
from src.ember.registry.model.schemas.provider_info import ProviderInfo
from src.ember.registry.model.schemas.cost import ModelCost, RateLimit


def test_model_registry_basic():
    registry = ModelRegistry()
    assert registry.list_models() == []

    provider_info = ProviderInfo(name="OpenAI", default_api_key="fake_key")
    model_info = ModelInfo(
        model_id="openai:gpt-4",
        model_name="gpt-4",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=provider_info,
        api_key="fake_key",
    )
    registry.register_model(model_info)
    assert "openai:gpt-4" in registry.list_models()

    model_instance = registry.get_model("openai:gpt-4")
    assert model_instance is not None


def test_model_registry_duplicate():
    registry = ModelRegistry()

    provider_info = ProviderInfo(name="OpenAI", default_api_key="fake_key")
    model_info = ModelInfo(
        model_id="openai:gpt-4",
        model_name="gpt-4",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=provider_info,
        api_key="fake_key",  # Ensure we have a valid key
    )

    # First registration is fine
    registry.register_model(model_info)

    # Attempt to re-register => ValueError
    with pytest.raises(ValueError, match="already registered"):
        registry.register_model(model_info)
