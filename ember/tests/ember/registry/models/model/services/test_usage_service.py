import pytest

from src.ember.registry.model.services.usage_service import UsageService


def test_usage_service_add_record():
    svc = UsageService()
    svc.add_usage_record(
        model_id="test-model",
        usage_stats={"total_tokens": 30, "prompt_tokens": 10, "completion_tokens": 20},
    )
    summary = svc.get_usage_summary("test-model")
    assert summary.total_tokens_used == 30


def test_usage_service_accumulate():
    svc = UsageService()
    svc.add_usage_record("acc-model", {"total_tokens": 10})
    svc.add_usage_record("acc-model", {"total_tokens": 25})

    summary = svc.get_usage_summary("acc-model")
    assert summary.total_tokens_used == 35
