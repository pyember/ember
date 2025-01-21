import pytest

from src.avior.registry.model.exceptions import ProviderConfigError
from src.avior.registry.model.registry.factory import ModelFactory
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from avior.registry.model.provider_registry.openai.openai_provider import OpenAIModel


def test_factory_create_openai_model():
    provider_info = ProviderInfo(name="OpenAI", default_api_key="fake_key")
    model_info = ModelInfo(
        model_id="openai:gpt-4",
        model_name="gpt-4",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=provider_info,
        api_key="fake_key",
    )
    model = ModelFactory.create_model_from_info(model_info)
    assert isinstance(model, OpenAIModel)


def test_factory_unsupported_provider():
    provider_info = ProviderInfo(name="NotAProvider", default_api_key="fake")
    model_info = ModelInfo(
        model_id="unknown:1",
        model_name="unknown:1",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=provider_info,
    )
    with pytest.raises(ProviderConfigError, match="Unrecognized model ID 'unknown:1'"):
        ModelFactory.create_model_from_info(model_info)
