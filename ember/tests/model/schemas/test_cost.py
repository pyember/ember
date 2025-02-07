import pytest
from pydantic import ValidationError

from src.ember.registry.model.schemas.cost import ModelCost, RateLimit


def test_model_cost_creation():
    cost = ModelCost(input_cost_per_thousand=0.01, output_cost_per_thousand=0.02)
    assert cost.input_cost_per_thousand == 0.01
    assert cost.output_cost_per_thousand == 0.02


def test_model_cost_negative():
    with pytest.raises(ValidationError):
        ModelCost(input_cost_per_thousand=-1.0, output_cost_per_thousand=0.01)


def test_rate_limit_creation():
    rl = RateLimit(tokens_per_minute=1000, requests_per_minute=50)
    assert rl.tokens_per_minute == 1000
    assert rl.requests_per_minute == 50


def test_rate_limit_negative():
    with pytest.raises(ValidationError):
        RateLimit(tokens_per_minute=-10, requests_per_minute=5)
