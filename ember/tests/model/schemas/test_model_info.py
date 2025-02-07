import pytest
from pydantic import ValidationError

from src.ember.registry.model.schemas.model_info import ModelInfo
from src.ember.registry.model.schemas.provider_info import ProviderInfo
from src.ember.registry.model.schemas.cost import ModelCost, RateLimit


def test_model_info_inherits_api_key():
    provider = ProviderInfo(name="OpenAI", default_api_key="default-key")
    mi = ModelInfo(
        model_id="openai:gpt-4",
        model_name="gpt-4",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=provider,
        api_key=None,
    )
    assert mi.api_key == "default-key"
    assert mi.get_api_key() == "default-key"


def test_model_info_no_api_key():
    provider = ProviderInfo(name="OpenAI", default_api_key=None)
    with pytest.raises(ValidationError):
        ModelInfo(
            model_id="openai:gpt-4",
            model_name="gpt-4",
            cost=ModelCost(),
            rate_limit=RateLimit(),
            provider=provider,
            api_key=None,
        )
