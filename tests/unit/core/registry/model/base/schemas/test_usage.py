"""Unit tests for UsageStats.
Verifies that usage statistics are aggregated correctly.
"""

import pytest

from ember.core.registry.model.base.schemas.usage import UsageStats


def test_usage_stats_addition() -> None:
    """Test that adding two UsageStats instances produces correct totals."""
    usage1 = UsageStats(
        total_tokens=100, prompt_tokens=60, completion_tokens=40, cost_usd=0.1
    )
    usage2 = UsageStats(
        total_tokens=150, prompt_tokens=90, completion_tokens=60, cost_usd=0.2
    )
    summed = usage1 + usage2
    assert summed.total_tokens == 250
    assert summed.prompt_tokens == 150
    assert summed.completion_tokens == 100
    assert summed.cost_usd == pytest.approx(0.3)


def test_usage_stats_add_method() -> None:
    """Test the add() method of UsageStats."""
    usage1 = UsageStats(
        total_tokens=80, prompt_tokens=50, completion_tokens=30, cost_usd=0.05
    )
    usage2 = UsageStats(
        total_tokens=20, prompt_tokens=10, completion_tokens=10, cost_usd=0.02
    )
    summed = usage1.add(usage2)
    assert summed.total_tokens == 100
    assert summed.prompt_tokens == 60
    assert summed.completion_tokens == 40
    assert summed.cost_usd == pytest.approx(0.07)
