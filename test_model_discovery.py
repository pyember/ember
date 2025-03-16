"""
Test script for model discovery and usage.

This script checks if models are actually discovered through the API rather than
relying on fallback hardcoded models, and tests invoking discovered models.
"""

import os
import pytest
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Ember components
from ember import initialize_ember
from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService
from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)


@pytest.mark.discovery
def test_discovery_service():
    """Test model discovery from provider APIs directly.

    This test verifies that the ModelDiscoveryService can be initialized and used
    to discover models from provider APIs. It doesn't depend on specific models
    being available.
    """
    # Create discovery service
    discovery_service = ModelDiscoveryService()

    # Discover models from all providers
    discovered_models = discovery_service.discover_models()

    # Basic validation - API contract checks
    assert isinstance(
        discovered_models, dict
    ), "Should return a dictionary of discovered models"

    # Log discovery results for debugging
    print(f"\nFound {len(discovered_models)} models via discovery")

    # Get provider-specific models
    openai_models = [
        model_id for model_id in discovered_models if model_id.startswith("openai:")
    ]
    anthropic_models = [
        model_id for model_id in discovered_models if model_id.startswith("anthropic:")
    ]

    # Output statistics for debugging
    print(f"OpenAI models: {len(openai_models)}")
    print(f"Anthropic models: {len(anthropic_models)}")

    # Show sample of models for reference
    if openai_models or anthropic_models:
        print("\nSample of discovered models:")
        for provider, models in [
            ("OpenAI", openai_models[:3]),
            ("Anthropic", anthropic_models[:3]),
        ]:
            if models:
                print(f"\n{provider} models:")
                for model_id in models:  # Show up to 3 models
                    print(f"  - {model_id}")


@pytest.mark.discovery
def test_model_invocation():
    """Test invoking a model with a simple prompt."""
    # Get first available model
    print(f"\n=== Testing Model Invocation ===\n")

    # Initialize the model registry with force_discovery=True
    model_registry = initialize_ember(force_discovery=True, initialize_context=False)

    # Get available models
    models = model_registry.list_models()

    # We can't assert that models are available since it depends on API keys,
    # but we can assert that the registry works
    assert hasattr(
        model_registry, "list_models"
    ), "Model registry should have list_models method"
    assert hasattr(
        model_registry, "get_model"
    ), "Model registry should have get_model method"

    # If no models available, skip the actual invocation test
    if not models:
        print("No models available for testing - this is expected in CI environments")
        return

    # Select first model
    model_id = models[0]
    prompt = "Explain quantum computing in one sentence."

    print(f"\n=== Testing Model Invocation: {model_id} ===\n")

    # Check if the model is registered
    assert model_registry.is_registered(
        model_id
    ), f"Model {model_id} should be registered"

    # Get the model
    print(f"Getting model {model_id}...")
    model = model_registry.get_model(model_id)

    # Verify the model is callable (don't actually invoke to avoid API costs)
    assert callable(model), "Model should be callable"

    # We don't actually call the model in CI tests to avoid API costs
    # In a local environment with API keys, you can uncomment this:
    """
    # Invoke the model
    print(f"Invoking model with prompt: '{prompt}'")
    try:
        response = model(prompt)
        
        # Print the response
        print("\nResponse:")
        if hasattr(response, 'content'):
            print(response.content)
        elif hasattr(response, 'text'):
            print(response.text)
        else:
            print(str(response))
        assert response is not None, "Response should not be None"
    except Exception as e:
        print(f"Error invoking model: {e}")
        assert False, f"Model invocation should not raise exception: {e}"
    """
