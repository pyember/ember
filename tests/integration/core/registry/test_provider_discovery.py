"""Integration tests for model provider discovery.

These tests verify actual API interactions with provider discovery mechanisms.
They only run when explicitly enabled via environment variables.
"""

import os
import pytest
import time

from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)
from ember.core.registry.model.providers.deepmind.deepmind_discovery import (
    DeepmindDiscovery,
)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_PROVIDER_INTEGRATION_TESTS"),
    reason="Provider integration tests only run when explicitly enabled",
)
class TestProviderDiscoveryIntegration:
    """Integration tests for provider discovery mechanisms.

    Enable with: RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/core/registry/test_provider_discovery.py -v
    """

    def check_minimal_model_data(self, model_data):
        """Verify the minimal structure of model metadata."""
        assert "model_id" in model_data
        assert "model_name" in model_data
        assert "api_data" in model_data

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY"
    )
    def test_openai_discovery_integration(self):
        """Test OpenAI discovery with actual API call."""
        discovery = OpenAIDiscovery()
        models = discovery.fetch_models()

        # Basic structure checks
        assert models, "No models returned from OpenAI discovery"
        assert any(
            "gpt-" in model_id for model_id in models.keys()
        ), "No GPT models found"

        # Check format of one model
        example_model = next(iter(models.values()))
        self.check_minimal_model_data(example_model)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY"
    )
    def test_anthropic_discovery_integration(self):
        """Test Anthropic discovery with actual API call."""
        discovery = AnthropicDiscovery()
        models = discovery.fetch_models()

        # Basic structure checks
        assert models, "No models returned from Anthropic discovery"
        assert any(
            "claude" in model_id for model_id in models.keys()
        ), "No Claude models found"

        # Check format of one model
        example_model = next(iter(models.values()))
        self.check_minimal_model_data(example_model)

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"), reason="Requires GOOGLE_API_KEY"
    )
    def test_deepmind_discovery_integration(self):
        """Test Deepmind/Google discovery with actual API call."""
        discovery = DeepmindDiscovery()
        models = discovery.fetch_models()

        # Basic structure checks (including fallback mechanism)
        assert models, "No models returned from Google/Deepmind discovery"
        assert any(
            "gemini" in model_id for model_id in models.keys()
        ), "No Gemini models found"

        # Check format of one model
        example_model = next(iter(models.values()))
        self.check_minimal_model_data(example_model)

    def test_model_registry_with_timeout(self):
        """Test that the ModelRegistry discovery has proper timeout handling."""
        import time

        # Create registry
        registry = ModelRegistry()

        # Time the discovery process
        start_time = time.time()
        registry.discover_models()
        elapsed_time = time.time() - start_time

        # Discovery with API calls should either:
        # 1. Return models within timeout period
        # 2. Return fallback models when timeout occurs
        assert len(registry.list_available_models()) > 0, "No models discovered"
        assert elapsed_time < 120, "Discovery took too long"
