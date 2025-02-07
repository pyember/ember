import pytest
from ember.registry.model_registry import ModelRegistry
from ember.registry.domain_models import UsageRecord


def test_model_registry_initialization(sample_model_infos):
    registry = ModelRegistry(sample_model_infos)
    model_ids = registry.list_models()
    # Should contain the model IDs from the fixture
    assert "gpt-4" in model_ids
    assert "claude-2" in model_ids


def test_model_registry_get_model(sample_model_infos):
    registry = ModelRegistry(sample_model_infos)
    model = registry.get_model("gpt-4")
    assert model is not None


def test_model_registry_add_usage_record(sample_model_infos):
    registry = ModelRegistry(sample_model_infos)
    record = UsageRecord(tokens_used=100)
    registry.add_usage_record("gpt-4", record)
    summary = registry.get_usage_summary("gpt-4")
    assert len(summary.records) == 1
    assert summary.records[0].tokens_used == 100


def test_model_registry_missing_model(sample_model_infos):
    registry = ModelRegistry(sample_model_infos)
    with pytest.raises(ValueError):
        registry.add_usage_record("nonexistent-id", UsageRecord(tokens_used=10))
