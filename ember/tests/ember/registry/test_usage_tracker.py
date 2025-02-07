import pytest
from ember.registry.usage_tracker import UsageTracker
from ember.registry.domain_models import UsageRecord


def test_register_model_usage():
    tracker = UsageTracker()
    tracker.register_model_usage("model-a")
    summary = tracker.get_usage_summary("model-a")
    assert summary is not None
    assert summary.model_name == "model-a"
    assert summary.window_duration == 60


def test_add_usage_record():
    tracker = UsageTracker()
    tracker.register_model_usage("model-b")
    tracker.add_usage_record("model-b", UsageRecord(tokens_used=123))
    summary = tracker.get_usage_summary("model-b")
    assert len(summary.records) == 1
    assert summary.records[0].tokens_used == 123


def test_usage_record_nonexistent_model():
    tracker = UsageTracker()
    with pytest.raises(ValueError) as exc:
        tracker.add_usage_record("missing-model", UsageRecord(tokens_used=10))
    assert "No usage summary found for model" in str(exc.value)
