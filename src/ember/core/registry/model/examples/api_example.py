"""Model API usage examples.

This module demonstrates how to use the high-level Ember API for model interactions.
"""

import logging

from ember.api import ModelAPI, ModelBuilder, ModelEnum, models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_namespace_access() -> None:
    """Demonstrate namespace-style access to models.

    This is the recommended approach for most use cases.
    """
    try:
        # Direct provider.model access
        response = models.openai.gpt4o("What is the capital of France?")
        print(f"OpenAI response: {response.data}")

        # Using aliases for common models
        response = models.gpt4("What is the capital of Germany?")
        print(f"GPT-4 response: {response.data}")

        # Using another provider
        response = models.anthropic.claude_3_5_sonnet("What is the capital of Italy?")
        print(f"Claude response: {response.data}")
    except Exception as error:
        logger.exception("Error with namespace access: %s", error)


def demo_builder_pattern() -> None:
    """Demonstrate builder pattern for model configuration."""
    try:
        # Create a configured model using the builder pattern
        model = (
            ModelBuilder()
            .temperature(0.7)
            .max_tokens(100)
            .timeout(30)
            .build(ModelEnum.OPENAI_GPT4O)
        )

        # Generate a response using the configured model
        response = model.generate(prompt="Explain quantum computing")
        print(f"Builder pattern response: {response.data}")
    except Exception as error:
        logger.exception("Error with builder pattern: %s", error)


def demo_enum_access() -> None:
    """Demonstrate enum-based access for type safety."""
    try:
        # Create a model API using enum for type safety
        model = ModelAPI.from_enum(ModelEnum.OPENAI_GPT4O)

        # Generate a response with additional parameters
        response = model.generate(
            prompt="What's your favorite programming language?",
            temperature=0.8,
            max_tokens=150,
        )
        print(f"Enum-based model response: {response.data}")
    except Exception as error:
        logger.exception("Error with enum access: %s", error)


def demo_registry_access() -> None:
    """Demonstrate direct registry access for advanced usage."""
    try:
        # Get the registry
        registry = models.get_registry()

        # List available models
        available_models = registry.list_models()
        print(f"Available models: {available_models}")

        # Get model info
        for model_id in available_models[:3]:  # Show first 3 models
            model_info = registry.get_model_info(model_id)
            print(f"Model: {model_id}")
            print(f"  Name: {model_info.model_name}")
            print(f"  Provider: {model_info.provider.name}")
            print(
                f"  Input Cost: ${model_info.cost.input_cost_per_thousand/1000:.6f} per token"
            )
            print(
                f"  Output Cost: ${model_info.cost.output_cost_per_thousand/1000:.6f} per token"
            )
            print()
    except Exception as error:
        logger.exception("Error with registry access: %s", error)


def demo_usage_tracking() -> None:
    """Demonstrate usage tracking."""
    try:
        # Get the usage service
        usage_service = models.get_usage_service()

        # Make some model calls
        models.gpt4("Tell me a short joke")
        models.claude("What's the weather like today?")

        # Get usage stats
        stats = usage_service.get_usage_stats()
        print(f"Total tokens used: {stats.total_tokens}")
        print(f"Total cost: ${stats.total_cost:.4f}")

        # Get usage by model
        for model, usage in stats.usage_by_model.items():
            print(f"Model {model}: {usage.total_tokens} tokens, ${usage.cost:.4f}")
    except Exception as error:
        logger.exception("Error with usage tracking: %s", error)


def main() -> None:
    """Run all demonstrations."""
    print("\n=== Namespace Access ===")
    demo_namespace_access()

    print("\n=== Builder Pattern ===")
    demo_builder_pattern()

    print("\n=== Enum Access ===")
    demo_enum_access()

    print("\n=== Registry Access ===")
    demo_registry_access()

    print("\n=== Usage Tracking ===")
    demo_usage_tracking()


if __name__ == "__main__":
    main()
