import pytest
from ember.registry.domain_models import ModelInfo, ProviderInfo, ModelCost, RateLimit
from ember.registry.model.factory import ModelFactory
from ember.registry.model import OpenAIModel, AnthropicModel, GeminiModel, WatsonXModel


@pytest.mark.parametrize(
    "provider_name, expected_class",
    [
        ("OpenAI", OpenAIModel),
        ("Anthropic", AnthropicModel),
        ("Google", GeminiModel),
        ("WatsonX", WatsonXModel),
    ],
)
def test_factory_creates_correct_model(provider_name, expected_class):
    info = ModelInfo(
        model_id="some_id",
        model_name="some_name",
        provider=ProviderInfo(name=provider_name),
        cost=ModelCost(),
        rate_limit=RateLimit(),
    )
    model = ModelFactory.create_model_from_info(info)
    assert isinstance(model, expected_class)


def test_factory_raises_for_unsupported_provider():
    info = ModelInfo(
        model_id="some_id",
        model_name="invalid_provider_name",
        provider=ProviderInfo(name="Unsupported"),
        cost=ModelCost(),
        rate_limit=RateLimit(),
    )
    with pytest.raises(ValueError) as exc:
        ModelFactory.create_model_from_info(info)
    assert "Unsupported provider" in str(exc.value)
