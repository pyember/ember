"""Tests for pricing configuration and updates."""

from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from ember.models.pricing import get_model_cost, get_model_costs
from ember.models.pricing_updater import PricingUpdater


class TestPricingYAML:
    """Test the pricing YAML configuration."""

    def test_pricing_yaml_exists(self):
        """Pricing YAML file should exist."""
        pricing_path = (
            Path(__file__).parent.parent.parent.parent / "src/ember/models/pricing/pricing.yaml"
        )
        assert pricing_path.exists(), f"pricing.yaml not found at {pricing_path}"

    def test_pricing_yaml_valid(self):
        """Pricing YAML should be valid and well-formed."""
        pricing_path = (
            Path(__file__).parent.parent.parent.parent / "src/ember/models/pricing/pricing.yaml"
        )

        with open(pricing_path) as f:
            data = yaml.safe_load(f)

        # Basic structure validation
        assert "version" in data
        assert "providers" in data
        assert isinstance(data["providers"], dict)

        # Validate each provider
        for provider, provider_data in data["providers"].items():
            assert "models" in provider_data, f"{provider} missing 'models'"
            assert isinstance(provider_data["models"], dict)

            # Validate each model
            for model, pricing in provider_data["models"].items():
                assert "input" in pricing, f"{provider}/{model} missing 'input'"
                assert "output" in pricing, f"{provider}/{model} missing 'output'"
                assert "context" in pricing, f"{provider}/{model} missing 'context'"

                # Validate types and ranges
                assert isinstance(pricing["input"], (int, float))
                assert isinstance(pricing["output"], (int, float))
                assert isinstance(pricing["context"], int)
                assert pricing["input"] >= 0
                assert pricing["output"] >= 0
                assert pricing["context"] > 0

    def test_known_models_have_pricing(self):
        """All known models should have pricing data."""
        known_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-3.5-turbo",
            "claude-4-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.5-haiku-latest",
            "gemini-1.5-pro",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ]

        costs = get_model_costs()

        for model in known_models:
            assert model in costs, f"Missing pricing for {model}"
            assert costs[model]["input"] > 0, f"Zero input cost for {model}"
            assert costs[model]["output"] > 0, f"Zero output cost for {model}"


class TestCostCalculation:
    """Test cost calculation functions."""

    def test_get_model_cost_known_model(self):
        """Test getting cost for a known model."""
        cost = get_model_cost("gpt-4")

        assert "input" in cost
        assert "output" in cost
        assert "context" in cost
        assert cost["input"] > 0
        assert cost["output"] > 0
        assert cost["context"] > 0

    def test_get_model_cost_unknown_model(self):
        """Unknown models should fail closed by default."""
        import pytest

        from ember.models.pricing.manager import PricingNotFoundError

        with pytest.raises(PricingNotFoundError):
            get_model_cost("unknown-model-xyz")

        cost = get_model_cost("unknown-model-xyz", strict=False)
        assert cost["input"] == 0.0
        assert cost["output"] == 0.0
        assert cost["context"] == 4096  # Default context

    def test_cost_calculation_accuracy(self):
        """Test that cost calculations are accurate."""
        # GPT-4: $30/1M input, $60/1M output (from YAML)
        cost = get_model_cost("gpt-4")

        # Calculate cost for 1000 input + 500 output tokens
        input_cost = (1000 / 1_000_000.0) * cost["input"]
        output_cost = (500 / 1_000_000.0) * cost["output"]
        total = round(input_cost + output_cost, 6)

        # Should be: 1 * 0.03 + 0.5 * 0.06 = 0.03 + 0.03 = 0.06
        assert total == 0.06

    def test_pricing_is_reasonable(self):
        """Test that pricing values are in reasonable ranges."""
        costs = get_model_costs()

        for model, cost in costs.items():
            # Input costs should be positive but reasonable
            assert (
                0 < cost["input"] < 1000.0
            ), f"{model} input cost seems wrong: ${cost['input']}/1M tokens"

            # Output can be 0 for non-generative endpoints (e.g. embeddings).
            assert 0.0 <= cost["output"] < 1000.0, (
                f"{model} output cost seems wrong: ${cost['output']}/1M tokens"
            )
            if cost["output"] > 0:
                assert cost["output"] >= cost["input"], f"{model} output should cost >= input"

            # Context should be reasonable
            assert cost["context"] >= 1024, f"{model} context too small: {cost['context']}"


class TestPricingUpdater:
    """Test the automated pricing updater."""

    def test_updater_initialization(self, tmp_path):
        """Test updater can be initialized."""
        yaml_path = tmp_path / "test_pricing.yaml"
        updater = PricingUpdater(yaml_path)

        assert updater.pricing_path == yaml_path
        assert len(updater.sources) == 3

    def test_compare_pricing_detects_changes(self):
        """Test that pricing comparison detects changes."""
        updater = PricingUpdater()

        old_data = {
            "providers": {
                "openai": {
                    "models": {
                        "gpt-4": {"input": 30.0, "output": 60.0},
                        "gpt-3.5": {"input": 0.5, "output": 1.5},
                    }
                }
            }
        }

        new_data = {
            "providers": {
                "openai": {
                    "models": {
                        "gpt-4": {"input": 25.0, "output": 50.0},  # Changed
                        "gpt-3.5": {"input": 0.5, "output": 1.5},  # Same
                        "gpt-4-turbo": {"input": 10.0, "output": 30.0},  # New
                    }
                }
            }
        }

        changes = updater._compare_pricing(old_data, new_data)

        assert len(changes) == 3
        assert any("input $30.0 → $25.0" in c for c in changes)
        assert any("output $60.0 → $50.0" in c for c in changes)
        assert any("NEW MODEL" in c for c in changes)

    def test_validate_pricing_catches_errors(self):
        """Test that validation catches malformed data."""
        updater = PricingUpdater()

        # Missing output price
        bad_data = {
            "providers": {"openai": {"models": {"gpt-4": {"input": 30.0}}}}  # Missing output
        }

        errors = updater.validate_pricing(bad_data)
        assert len(errors) == 1
        assert "Missing 'output' price" in errors[0]

    @patch("ember.models.pricing_updater.models")
    def test_fetch_provider_pricing(self, mock_models_func):
        """Test fetching pricing from a provider."""
        # Mock the models API response
        mock_response = Mock()
        mock_response.text = """Here's the pricing:

```yaml
gpt-4:
  input: 30.0
  output: 60.0
  context: 8192
gpt-3.5-turbo:
  input: 0.5
  output: 1.5
  context: 16385
```
"""
        # Set up the mock to return our response
        mock_models_func.return_value = mock_response

        updater = PricingUpdater()
        result = updater.fetch_provider_pricing("openai", "https://example.com")

        assert "gpt-4" in result
        assert result["gpt-4"]["input"] == 30.0
        assert result["gpt-4"]["output"] == 60.0

    def test_dry_run_does_not_write(self, tmp_path):
        """Test that dry run doesn't modify files."""
        yaml_path = tmp_path / "test_pricing.yaml"

        # Create initial file
        initial_data = {
            "version": "1.0",
            "providers": {"openai": {"models": {"gpt-4": {"input": 30.0, "output": 60.0}}}},
        }
        with open(yaml_path, "w") as f:
            yaml.dump(initial_data, f)

        # Run updater in dry-run mode
        updater = PricingUpdater(yaml_path)
        with patch.object(updater, "fetch_provider_pricing") as mock_fetch:
            # Mock different pricing
            mock_fetch.return_value = {"gpt-4": {"input": 25.0, "output": 50.0}}

            updater.update_pricing(dry_run=True)

        # File should be unchanged
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        assert data == initial_data
