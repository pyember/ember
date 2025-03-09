"""
Example demonstrating the Ember Models API.

This file shows how to use the models API to initialize and interact
with language models from different providers.

To run:
    poetry run python src/ember/examples/model_api_example.py
"""

import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from the simplified API
from ember.api.models import (
    initialize_registry,
    create_model_service,
    ModelRegistry,
    ModelService,
    UsageService,
    ModelInfo,
    ModelCost,
    RateLimit
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
    model_service = ModelService(
        registry=registry,
        usage_service=usage_service
    )
    
    # Using OpenAI's model
    try:
        response = model_service.invoke_model(
            model_id="openai:gpt-4",
            prompt="What is the capital of France?",
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"Model: {response.model}")
        print(f"Response: {response.text[:150]}...")  # Show truncated response for readability
        
        if hasattr(response, "usage") and response.usage:
            print(f"Token usage: {response.usage.total_tokens} tokens")
            
    except Exception as e:
        print(f"Error using model service: {e}")
    
    return model_service, usage_service


def direct_model_example(registry):
    """Demonstrate getting and using a model directly."""
    print("\n=== Direct Model Example ===")
    
    try:
        # Get a model instance directly from the registry
        model = registry.get_model("anthropic:claude-3-sonnet")
        
        if model:
            # Call the model directly
            response = model(
                prompt="Explain quantum computing in simple terms.",
                temperature=0.5,
                max_tokens=150
            )
            
            print(f"Model: {model.model_info.id}")
            print(f"Response: {response.text[:150]}...")  # Show truncated response for readability
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
            if registry.model_exists(model_id):
                info = registry.get_model_info(model_id)
                print(f"\nModel: {model_id}")
                print(f"  Name: {info.name}")
                print(f"  Provider: {info.provider.get('name', 'Unknown')}")
                print(f"  Context window: {info.context_window} tokens")
                print(f"  Input cost: ${info.cost.input_cost_per_thousand:.4f} per 1K tokens")
                print(f"  Output cost: ${info.cost.output_cost_per_thousand:.4f} per 1K tokens")
            else:
                print(f"Model {model_id} not found in registry")
    except Exception as e:
        print(f"Error accessing model metadata: {e}")


def usage_tracking_example(model_service, usage_service):
    """Demonstrate usage tracking capabilities."""
    print("\n=== Usage Tracking Example ===")
    
    try:
        # Make a few model calls to generate usage data
        prompts = [
            "Write a short poem about programming.",
            "Explain the difference between supervised and unsupervised learning."
        ]
        
        for prompt in prompts:
            try:
                model_service.invoke_model(
                    model_id="openai:gpt-4",
                    prompt=prompt,
                    max_tokens=100
                )
                print(f"Successfully called model with prompt: {prompt[:30]}...")
            except Exception as e:
                print(f"Error making model call: {e}")
        
        # Get usage statistics
        stats = usage_service.get_stats()
        print("\nUsage Summary:")
        print(f"  Total requests: {stats.total_requests}")
        print(f"  Total tokens: {stats.total_input_tokens + stats.total_output_tokens}")
        print(f"  Estimated cost: ${stats.total_cost:.4f}")
        
        # Get cost breakdown by model
        costs = usage_service.get_costs_by_model()
        if costs:
            print("\nCosts by Model:")
            for model_id, cost in costs.items():
                print(f"  {model_id}: ${cost:.4f}")
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
                input_cost_per_thousand=0.0005,
                output_cost_per_thousand=0.0015
            ),
            rate_limit=RateLimit(
                tokens_per_minute=100000,
                requests_per_minute=3000
            ),
            context_window=32000,
            provider={
                "name": "CustomAI",
                "api_base": "https://api.custom-ai.example.com/v1",
                "default_api_key": "${CUSTOM_API_KEY}"
            }
        )
        
        # Register the model
        registry.register_model(model_info=custom_model)
        print(f"Model {custom_model.id} registered successfully")
        
        # Verify it's in the registry
        if registry.model_exists(custom_model.id):
            info = registry.get_model_info(custom_model.id)
            print(f"Confirmed model in registry: {info.id} ({info.name})")
            print(f"Context window: {info.context_window} tokens")
        else:
            print("Custom model registration failed")
    except Exception as e:
        print(f"Error registering custom model: {e}")


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