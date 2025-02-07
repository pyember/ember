import pytest
from src.ember.domain.models import ModelInfo, ModelCost, RateLimit, ProviderInfo


@pytest.fixture
def sample_model_infos():
    return [
        ModelInfo(
            model_id="gpt-4",
            model_name="gpt-4",
            provider=ProviderInfo(name="OpenAI", default_api_key="sk-fake-openai"),
            cost=ModelCost(input_cost_per_thousand=0.03, output_cost_per_thousand=0.06),
            rate_limit=RateLimit(tokens_per_minute=80000, requests_per_minute=5000),
        ),
        ModelInfo(
            model_id="claude-2",
            model_name="claude-2",
            provider=ProviderInfo(
                name="Anthropic", default_api_key="sk-fake-anthropic"
            ),
            cost=ModelCost(
                input_cost_per_thousand=0.002, output_cost_per_thousand=0.01
            ),
            rate_limit=RateLimit(tokens_per_minute=400000, requests_per_minute=4000),
        ),
    ]
