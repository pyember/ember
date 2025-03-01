"""Unit tests for cost schemas (ModelCost and RateLimit).
"""

import pytest

from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit


def test_model_cost_non_negative() -> None:
    """Test that ModelCost rejects negative values."""
    with pytest.raises(ValueError):
        ModelCost(input_cost_per_thousand=-1, output_cost_per_thousand=2.0)


def test_rate_limit_non_negative() -> None:
    """Test that RateLimit rejects negative values."""
    with pytest.raises(ValueError):
        RateLimit(tokens_per_minute=-100, requests_per_minute=50)
