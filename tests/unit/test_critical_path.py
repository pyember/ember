"""Critical path tests."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from ember._internal.exceptions import ProviderAPIError
from ember.api import models
from ember.models import ModelRegistry
from ember.models.pricing import get_model_cost
from ember.models.schemas import ChatResponse, UsageStats


class TestCriticalPath:
    """Test the critical paths that actually matter to users."""

    def test_model_invocation_performance(self):
        """The ONE performance test that matters - user-facing latency."""
        # Mock a fast response
        mock_response = ChatResponse(
            data="Test response", usage=UsageStats(prompt_tokens=10, completion_tokens=20)
        )

        with patch("ember.api.models.ModelRegistry.invoke_model", return_value=mock_response):
            start = time.perf_counter()
            response = models("gpt-4", "Hello")
            elapsed = time.perf_counter() - start

            # Our SLA: <100ms for mocked calls
            assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s"
            assert response.text == "Test response"

    def test_pricing_invariants(self):
        """Costs can never be negative or absurdly high."""
        from ember.models.pricing import _pricing

        for model in _pricing.list_models():
            cost = get_model_cost(model)

            # Sanity bounds - no model costs $1000 per 1M tokens
            assert 0 <= cost["input"] < 1000, f"{model}: input cost out of bounds"
            assert 0 <= cost["output"] < 1000, f"{model}: output cost out of bounds"

            # Output usually costs more than input (except for some models)
            # This is a soft check - some models might violate this
            if cost["output"] < cost["input"]:
                # Log but don't fail - some models have equal costs
                print(f"Note: {model} has output < input cost")

    def test_concurrent_pricing_lookups(self):
        """Registry should handle concurrent access safely."""

        def get_cost():
            return get_model_cost("gpt-4")["input"]

        # Hit it with concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: get_cost(), range(100)))

        # All results should be identical
        unique_results = set(results)
        assert len(unique_results) == 1, f"Inconsistent results: {unique_results}"

    def test_corrupted_pricing_yaml(self):
        """Pricing lookups should fail closed by default."""
        from ember.models import pricing

        # Simulate corrupted YAML - patch the _load_yaml method directly
        with patch.object(pricing._pricing, "_load_yaml", side_effect=Exception("File corrupted")):
            # Unknown models should raise by default
            from ember.models.pricing.manager import PricingNotFoundError

            with pytest.raises(PricingNotFoundError):
                get_model_cost("unknown-model-xyz")

            # Opt-in legacy behavior still returns defaults
            cost = get_model_cost("unknown-model-xyz", strict=False)
            assert cost["input"] == 0.0
            assert cost["output"] == 0.0

    def test_does_it_work_reliably(self):
        """The only test that matters - does the whole system work?"""
        # Mock a complete response with all fields
        mock_response = ChatResponse(
            data="The answer is 42",
            usage=UsageStats(
                prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.0015
            ),
            model_id="gpt-4",
        )

        registry = ModelRegistry()

        with patch.object(registry, "get_model") as mock_get_model:
            # Mock the model
            mock_model = Mock()
            mock_model.complete.return_value = mock_response
            mock_get_model.return_value = mock_model

            # 1. It works
            response = registry.invoke_model("gpt-4", "What is the answer?")
            assert response.data == "The answer is 42"

            # 2. It tracks costs
            assert response.usage is not None
            assert 0 < response.usage.cost_usd < 1.0

            # 3. It handles errors gracefully
            mock_model.complete.side_effect = Exception("API Error")
            with pytest.raises(ProviderAPIError):
                registry.invoke_model("gpt-4", "This will fail")


class TestCostReconciliation:
    """Test that our pricing matches reality."""

    def test_cost_tracking_accuracy(self):
        """Verify cost tracking works correctly."""
        from ember.models.pricing.tracker import CostTracker

        tracker = CostTracker()

        # Simulate usage with exact match
        usage1 = UsageStats(
            prompt_tokens=100, completion_tokens=200, cost_usd=0.009, actual_cost_usd=0.009
        )
        tracker.record_usage(usage1, "gpt-4")

        # Simulate usage with small discrepancy
        usage2 = UsageStats(
            prompt_tokens=100,
            completion_tokens=200,
            cost_usd=0.009,
            actual_cost_usd=0.0095,  # 5.5% difference
        )
        tracker.record_usage(usage2, "gpt-4")

        metrics = tracker.get_accuracy_metrics()

        assert metrics["reconciliation_count"] == 2
        assert 90 < metrics["accuracy_pct"] < 100  # Should be ~97%
        assert metrics["max_deviation_usd"] < 0.001

    @pytest.mark.integration
    def test_real_api_cost_tracking(self):
        """Actually verify our pricing matches OpenAI's reality."""

        # This test requires a valid OpenAI API key with cost headers
        # Currently skipped as it needs special API access
        pytest.skip("Integration test - requires valid OpenAI API key with actual cost headers")

        # Use a cheap model for testing
        response = models("gpt-3.5-turbo", "Hi")

        if response.usage and response.usage.actual_cost_usd:
            # If OpenAI provides actual cost, verify it's close
            our_cost = response.usage.cost_usd
            actual_cost = response.usage.actual_cost_usd

            # Should be within 10% (allowing for rounding)
            if actual_cost > 0:
                accuracy = abs(our_cost - actual_cost) / actual_cost
                assert accuracy < 0.1, f"Cost mismatch: ours=${our_cost}, actual=${actual_cost}"

    def test_cost_accuracy_clamped(self):
        """Accuracy percentage should stay within [0, 100]."""
        from ember.models.pricing.tracker import CostTracker
        from ember.models.schemas import UsageStats

        tracker = CostTracker()
        # Large overestimate vs actual
        tracker.record_usage(
            UsageStats(prompt_tokens=0, completion_tokens=0, cost_usd=100.0, actual_cost_usd=1.0),
            "gpt-4",
        )
        metrics = tracker.get_accuracy_metrics()
        assert 0.0 <= metrics.get("accuracy_pct", 0.0) <= 100.0


if __name__ == "__main__":
    # Run with: python test_critical_path.py
    pytest.main([__file__, "-v"])
