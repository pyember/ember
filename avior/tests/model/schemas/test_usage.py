import pytest
from src.avior.registry.model.schemas.usage import UsageRecord, UsageSummary


def test_usage_record_creation():
    ur = UsageRecord(
        usage_stats={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}
    )
    assert ur.usage_stats.total_tokens == 100
    assert ur.usage_stats.prompt_tokens == 50
    assert ur.usage_stats.completion_tokens == 50


def test_usage_summary_add_record():
    summary = UsageSummary(model_name="test-model")
    assert summary.total_tokens_used == 0

    rec = UsageRecord(usage_stats={"total_tokens": 10})
    summary.add_record(rec)
    assert summary.total_tokens_used == 10

    rec2 = UsageRecord(usage_stats={"total_tokens": 5})
    summary.add_record(rec2)
    assert summary.total_tokens_used == 15
