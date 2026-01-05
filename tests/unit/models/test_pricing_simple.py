"""Tests for the simplified pricing system."""

from ember.models.pricing import Pricing


class TestPricingClass:
    """Test the new simple Pricing class."""

    def test_pricing_loads_yaml(self):
        """Test that pricing loads from YAML."""
        pricing = Pricing()

        # Should have loaded some models
        models = pricing.list_models()
        assert len(models) > 0
        assert "gpt-4" in models
        assert "claude-3-opus" in models

    def test_calculate_cost(self):
        """Test cost calculation."""
        pricing = Pricing()

        # GPT-4: $30 per 1M input, $60 per 1M output
        cost = pricing.calculate_cost("gpt-4", input_tokens=1000, output_tokens=500)

        # Should be: (1000 * 30 + 500 * 60) / 1_000_000 = 0.06
        assert cost == 0.06

    def test_unknown_model_returns_zero(self):
        """Test that unknown models return zero cost in non-strict mode."""
        pricing = Pricing(strict=False)

        cost = pricing.calculate_cost("unknown-model", input_tokens=1000, output_tokens=1000)
        assert cost == 0.0

    def test_unknown_model_raises_in_strict_mode(self):
        """Test that unknown models raise PricingNotFoundError in strict mode."""
        import pytest

        from ember.models.pricing.manager import PricingNotFoundError

        pricing = Pricing(strict=True)

        with pytest.raises(PricingNotFoundError) as exc_info:
            pricing.calculate_cost("unknown-model", input_tokens=1000, output_tokens=1000)

        assert exc_info.value.model_id == "unknown-model"

    def test_get_model_pricing(self):
        """Test getting pricing for a specific model."""
        pricing = Pricing()

        gpt4_pricing = pricing.get_model_pricing("gpt-4")
        assert gpt4_pricing["input"] == 30.0  # Per million
        assert gpt4_pricing["output"] == 60.0  # Per million
        assert gpt4_pricing["context"] == 8192

    def test_pricing_with_custom_yaml(self, tmp_path):
        """Test loading from custom YAML path."""
        import yaml

        # Create test YAML
        test_yaml = tmp_path / "test_pricing.yaml"
        test_data = {
            "version": "1.0",
            "providers": {
                "test": {"models": {"test-model": {"input": 10.0, "output": 20.0, "context": 4096}}}
            },
        }

        with open(test_yaml, "w") as f:
            yaml.dump(test_data, f)

        # Load from custom path
        pricing = Pricing(test_yaml)

        assert pricing.list_models() == ["test-model"]
        assert pricing.get_model_pricing("test-model")["input"] == 10.0

    def test_large_token_calculation(self):
        """Test calculation with large token counts."""
        pricing = Pricing()

        # 1M tokens each
        cost = pricing.calculate_cost("gpt-4", input_tokens=1_000_000, output_tokens=1_000_000)

        # Should be: 30 + 60 = $90
        assert cost == 90.0

    def test_fractional_costs(self):
        """Test with models that have fractional costs."""
        pricing = Pricing()

        # Claude Haiku: $0.25 input, $1.25 output per million
        cost = pricing.calculate_cost(
            "claude-3-haiku",
            input_tokens=100_000,
            output_tokens=100_000,  # 0.1M  # 0.1M
        )

        # Should be: 0.1 * 0.25 + 0.1 * 1.25 = 0.15
        assert cost == 0.15

