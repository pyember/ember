"""Integration tests for model provider discovery.

These tests verify actual API interactions with provider discovery mechanisms.
They only run when explicitly enabled via environment variables.
"""

import os
import time

import pytest

from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)
from ember.core.registry.model.providers.deepmind.deepmind_discovery import (
    DeepmindDiscovery,
)
from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_PROVIDER_INTEGRATION_TESTS"),
    reason="Provider integration tests only run when explicitly enabled with RUN_PROVIDER_INTEGRATION_TESTS=1",
)
class TestProviderDiscoveryIntegration:
    """Integration tests for provider discovery mechanisms.

    Enable with: RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/core/registry/test_provider_discovery.py -v
    """

    def check_minimal_model_data(self, model_data):
        """Verify the minimal structure of model metadata."""
        assert "id" in model_data
        assert "name" in model_data
        assert "api_data" in model_data

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY"
    )
    def test_openai_discovery_integration(self):
        """Test OpenAI discovery with actual API call."""
        api_key = os.environ.get("OPENAI_API_KEY")
        discovery = OpenAIDiscovery(api_key=api_key)
        models = discovery.fetch_models()

        # Basic structure checks - only runs if discovery succeeds
        if models:
            assert len(models) > 0, "No models found"
            # More permissive check that looks for any model with the openai prefix
            assert all(
                model_id.startswith("openai:") for model_id in models.keys()
            ), "Model IDs do not follow expected pattern"
        else:
            pytest.skip(
                "No models returned from OpenAI discovery - API may be unreachable"
            )

        # Check format of one model if models were found
        if models:
            example_model = next(iter(models.values()))
            self.check_minimal_model_data(example_model)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY"
    )
    def test_anthropic_discovery_integration(self):
        """Test Anthropic discovery with actual API call."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        discovery = AnthropicDiscovery(api_key=api_key)
        models = discovery.fetch_models()

        # Basic structure checks - only runs if discovery succeeds
        if models:
            assert len(models) > 0, "No models found"
            # More permissive check that looks for any model with the anthropic prefix
            assert all(
                model_id.startswith("anthropic:") for model_id in models.keys()
            ), "Model IDs do not follow expected pattern"
        else:
            pytest.skip(
                "No models returned from Anthropic discovery - API may be unreachable"
            )

        # Check format of one model if models were found
        if models:
            example_model = next(iter(models.values()))
            self.check_minimal_model_data(example_model)

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"),
        reason="Requires GOOGLE_API_KEY or GEMINI_API_KEY",
    )
    def test_deepmind_discovery_integration(self):
        """Test Deepmind/Google discovery with actual API call."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        discovery = DeepmindDiscovery()
        discovery.configure(api_key=api_key)
        models = discovery.fetch_models()

        # Basic structure checks - only runs if discovery succeeds
        if models:
            assert len(models) > 0, "No models found"
            # More permissive check that looks for any model with the deepmind prefix
            assert all(
                model_id.startswith("deepmind:") for model_id in models.keys()
            ), "Model IDs do not follow expected pattern"
        else:
            pytest.skip(
                "No models returned from Google/Deepmind discovery - API may be unreachable"
            )

        # Check format of one model if models were found
        if models:
            example_model = next(iter(models.values()))
            self.check_minimal_model_data(example_model)

    def test_model_registry_with_timeout(self):
        """Test that the ModelRegistry discovery has proper timeout handling."""

        # Create registry
        registry = ModelRegistry()

        # Time the discovery process
        start_time = time.time()
        registry.discover_models()
        elapsed_time = time.time() - start_time

        # Discovery should complete within a reasonable timeout
        # Note: May return empty results if API keys aren't set or discovery fails
        # This is expected behavior with fallbacks removed
        assert elapsed_time < 120, "Discovery took too long"

        # Log discovered models for debugging
        discovered_models = registry.list_models()
        if discovered_models:
            print(f"Discovered {len(discovered_models)} models")
        else:
            print("No models discovered - this is expected if no API keys are set")
