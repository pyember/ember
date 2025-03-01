"""Unit tests for the UsageService functionality.

Tests recording and aggregation of usage statistics and thread safety.
"""

import threading

import pytest

from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.base.schemas.usage import UsageStats


def test_add_and_get_usage() -> None:
    """Test that usage records are correctly added and aggregated."""
    usage_service = UsageService()
    model_id = "test:model"
    usage1 = UsageStats(
        total_tokens=100, prompt_tokens=60, completion_tokens=40, cost_usd=0.1
    )
    usage2 = UsageStats(
        total_tokens=200, prompt_tokens=120, completion_tokens=80, cost_usd=0.2
    )
    usage_service.add_usage_record(model_id=model_id, usage_stats=usage1)
    usage_service.add_usage_record(model_id=model_id, usage_stats=usage2)
    summary = usage_service.get_usage_summary(model_id=model_id)
    assert summary.total_usage.total_tokens == 300
    assert summary.total_usage.cost_usd == pytest.approx(0.3)


def test_thread_safety() -> None:
    """Test that UsageService is thread-safe by concurrently adding usage records."""
    usage_service = UsageService()
    model_id = "test:model"

    def add_usage() -> None:
        usage = UsageStats(
            total_tokens=10, prompt_tokens=5, completion_tokens=5, cost_usd=0.01
        )
        usage_service.add_usage_record(model_id=model_id, usage_stats=usage)

    threads = [threading.Thread(target=add_usage) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = usage_service.get_usage_summary(model_id=model_id)
    assert summary.total_usage.total_tokens == 10 * 50
