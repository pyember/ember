"""
Example demonstrating the Ember Models API.

This file shows how to use the models API to initialize and interact
with language models from different providers.

To run:
    uv run python src/ember/examples/models/model_api_example.py

    # Or if in an activated virtual environment
    python src/ember/examples/models/model_api_example.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key for OpenAI model examples
    ANTHROPIC_API_KEY (optional): Your Anthropic API key for Anthropic model examples
"""

import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from the simplified API
from ember.api.models import (
    ModelCost,
    ModelInfo,
    ModelService,
    RateLimit,
    UsageService,
    initialize_registry,
)


def registry_setup_example():
    """Demonstrate how to initialize the model registry."""
    print("\n=== Registry Initialization Example ===")

    # Initialize the registry with auto-discovery
    registry = initialize_registry(auto_discover=True)

    # List all discovered models
    model_ids = registry.list_models()
    print(f"Discovered {len(model_ids)} models:")

    # Show first 5 models (if available)
    for model_id in model_ids[:5]:
        print(f"  - {model_id}")

    if len(model_ids) > 5:
        print(f"  ... and {len(model_ids) - 5} more")

    return registry


def model_service_example(registry):
    """Demonstrate using ModelService to invoke models."""
    print("\n=== Model Service Example ===")

    # Create a usage service for tracking
    usage_service = UsageService()

    # Create a model service with the registry and usage tracking
    model_service = ModelService(registry=registry, usage_service=usage_service)

    # Using OpenAI's model
    try:
        if registry.is_registered("openai:gpt-4"):
            info = registry.get_model_info("openai:gpt-4")
            print(f"Model info found for openai:gpt-4: {info.name}")

            # Get model directly using fixed parse_model_str
            model = registry.get_model("openai:gpt-4")
            if model:
                print(f"Successfully retrieved model: {model.model_info.id}")

                # We would invoke it like this in a real application
                print("In a real application, you would invoke the model with:")
                print(
                    "response = model_service.invoke_model('openai:gpt-4', 'What is the capital of France?')"
                )

                # Avoiding actual API call to save credits
                print("(API call skipped in this example to avoid using API credits)")
            else:
                print("Model retrieval failed.")
        else:
            print("Model not found in registry.")

    except Exception as e:
        print(f"Error using model service: {e}")

    return model_service, usage_service


def direct_model_example(registry):
    """Demonstrate getting and using a model directly."""
    print("\n=== Direct Model Example ===")

    try:
        # Check if model is registered
        if registry.is_registered("anthropic:claude-3-sonnet"):
            info = registry.get_model_info("anthropic:claude-3-sonnet")
            print(f"Model info found for anthropic:claude-3-sonnet: {info.name}")

            # Get model directly
            model = registry.get_model("anthropic:claude-3-sonnet")
            if model:
                print(f"Successfully retrieved model: {model.model_info.id}")

                # We could call the model directly like this
                print("In a real application, you would call the model directly with:")
                print("response = model('Explain quantum computing in simple terms.')")

                # Avoiding actual API call to save credits
                print("(API call skipped in this example to avoid using API credits)")
            else:
                print("Model retrieval failed.")
        else:
            print("Model not found in registry.")
    except Exception as e:
        print(f"Error using model directly: {e}")


def model_metadata_example(registry):
    """Demonstrate accessing model metadata."""
    print("\n=== Model Metadata Example ===")

    try:
        # Get metadata for specific models
        for model_id in ["openai:gpt-4", "anthropic:claude-3-sonnet"]:
            if registry.is_registered(model_id):
                info = registry.get_model_info(model_id)
                if info:
                    print(f"\nModel: {model_id}")
                    print(f"  Name: {info.name}")
                    print(f"  Provider: {info.provider.name}")
                    print(
                        f"  Input cost: ${info.cost.input_cost_per_thousand:.4f} per 1K tokens"
                    )
                    print(
                        f"  Output cost: ${info.cost.output_cost_per_thousand:.4f} per 1K tokens"
                    )
            else:
                print(f"Model {model_id} not found in registry")
    except Exception as e:
        print(f"Error accessing model metadata: {e}")


def usage_tracking_example(model_service, usage_service):
    """Demonstrate usage tracking capabilities."""
    print("\n=== Usage Tracking Example ===")

    try:
        # Demonstration of usage tracking
        print("Usage tracking works with model invocations:")
        print("1. Make model calls with model_service.invoke_model()")
        print("2. UsageService automatically records token usage")
        print("3. Get usage statistics with usage_service.get_usage_summary()")

        # In a real application with actual API calls, you would see non-zero usage
        print(
            "\nIn this example, we skip actual API calls but demonstrate the tracking API"
        )

        # Retrieve usage summary for demonstration
        model_id = "openai:gpt-4"
        usage_summary = usage_service.get_usage_summary(model_id=model_id)

        print("\nUsage Summary:")
        print(f"  Model: {usage_summary.model_name}")
        print(f"  Total tokens: {usage_summary.total_tokens_used}")
        print(f"  Prompt tokens: {usage_summary.total_usage.prompt_tokens}")
        print(f"  Completion tokens: {usage_summary.total_usage.completion_tokens}")
        print(f"  Estimated cost: ${usage_summary.total_usage.cost_usd:.4f}")

        # Get summaries for all registered models
        print("\nAll Models Usage:")
        # Get the model registry's registered models
        registry = model_service._registry
        for model_id in registry.list_models():
            try:
                summary = usage_service.get_usage_summary(model_id=model_id)
                if summary.total_tokens_used > 0:
                    print(
                        f"  {model_id}: {summary.total_tokens_used} tokens, ${summary.total_usage.cost_usd:.4f}"
                    )
            except Exception as e:
                print(f"  Error getting usage for {model_id}: {e}")

    except Exception as e:
        print(f"Error tracking usage: {e}")


def custom_model_example(registry):
    """Demonstrate registering a custom model."""
    print("\n=== Custom Model Registration Example ===")

    try:
        # Create custom model info
        custom_model = ModelInfo(
            id="custom:my-custom-model",
            name="My Custom LLM",
            cost=ModelCost(
                input_cost_per_thousand=0.0005, output_cost_per_thousand=0.0015
            ),
            rate_limit=RateLimit(tokens_per_minute=100000, requests_per_minute=3000),
            provider={
                "name": "CustomAI",
                "base_url": "https://api.custom-ai.example.com/v1",
                "default_api_key": "${CUSTOM_API_KEY}",
            },
        )

        # Register the model
        registry.register_model(model_info=custom_model)
        print(f"Model {custom_model.id} registered successfully")

        # Verify it's in the registry
        if registry.is_registered(custom_model.id):
            info = registry.get_model_info(custom_model.id)
            print(f"Confirmed model in registry: {info.id} ({info.name})")
            print(f"Provider: {info.provider.name}")
        else:
            print("Custom model registration failed")
    except Exception as e:
        print(f"Error registering custom model: {e}")


def register_models(registry):
    """Register test models with the registry."""
    registered_count = 0

    # Register OpenAI models
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        # Create the models
        model_infos = [
            ModelInfo(
                id="openai:gpt-4",
                name="GPT-4",
                context_window=128000,
                cost=ModelCost(
                    input_cost_per_thousand=0.01, output_cost_per_thousand=0.03
                ),
                rate_limit=RateLimit(
                    tokens_per_minute=10000000, requests_per_minute=1000
                ),
                provider={
                    "name": "OpenAI",
                    "default_api_key": openai_key,
                    "base_url": "https://api.openai.com/v1",
                },
            )
        ]

        # Register the models, but check if they're already registered first
        for model_info in model_infos:
            if not registry.is_registered(model_info.id):
                registry.register_model(model_info=model_info)
                print(f"Registered model: {model_info.id}")
                registered_count += 1
            else:
                print(
                    f"Model {model_info.id} already registered ✅ - using existing registration"
                )

    # Register Anthropic models
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        model_infos = [
            ModelInfo(
                id="anthropic:claude-3-sonnet",
                name="Claude 3 Sonnet",
                context_window=200000,
                cost=ModelCost(
                    input_cost_per_thousand=0.003, output_cost_per_thousand=0.015
                ),
                rate_limit=RateLimit(
                    tokens_per_minute=5000000, requests_per_minute=1000
                ),
                provider={
                    "name": "Anthropic",
                    "default_api_key": anthropic_key,
                    "base_url": "https://api.anthropic.com/v1",
                },
            )
        ]

        # Register the models, but check if they're already registered first
        for model_info in model_infos:
            if not registry.is_registered(model_info.id):
                registry.register_model(model_info=model_info)
                print(f"Registered model: {model_info.id}")
                registered_count += 1
            else:
                print(
                    f"Model {model_info.id} already registered ✅ - using existing registration"
                )

    # Return the number of newly registered models
    return registered_count


def main():
    """Run all examples in sequence."""
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. OpenAI examples will fail.")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Anthropic examples will fail.")

    print("Running Models API examples...")

    # Example 1: Initialize registry
    registry = registry_setup_example()

    # Register models manually
    print("\nRegistering models manually:")
    register_models(registry)

    # Example 2: Model service
    model_service, usage_service = model_service_example(registry)

    # Example 3: Direct model access
    direct_model_example(registry)

    # Example 4: Model metadata
    model_metadata_example(registry)

    # Example 5: Usage tracking
    usage_tracking_example(model_service, usage_service)

    # Example 6: Custom model
    custom_model_example(registry)

    print("\nAll examples completed.")


if __name__ == "__main__":
    main()
