import pytest
from avior.registry.domain_models import ModelInfo, ProviderInfo, ModelCost, RateLimit
from avior.registry.model import OpenAIModel


@pytest.fixture
def openai_model():
    info = ModelInfo(
        model_id="gpt-4",
        model_name="gpt-4",
        provider=ProviderInfo(name="OpenAI", default_api_key="fake-key"),
        cost=ModelCost(),
        rate_limit=RateLimit(),
    )
    return OpenAIModel(info)


def test_openai_forward_success(openai_model):
    resp = openai_model.forward("Hello from test suite")
    assert "GPT-4 response" in resp


def test_openai_forward_empty_query(openai_model):
    with pytest.raises(ValueError):
        openai_model.forward("")


def test_openai_usage_calculation(openai_model):
    _ = openai_model.forward("Hello from test suite")
    usage_data = openai_model.calculate_usage(openai_model.last_raw_output)
    assert usage_data["total_tokens"] == 50
    assert usage_data["prompt_tokens"] == 25
    assert usage_data["completion_tokens"] == 25
