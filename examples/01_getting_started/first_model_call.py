"""Make your first LLM API call with Ember.

This example demonstrates:
- Basic model invocation with ember.models()
- Getting structured responses with models.response()
- Discovering available models
- Graceful handling when API keys are not configured

Prerequisites:
    Configure at least one provider:
        ember setup
    Or:
        ember configure set providers.openai.api_key "sk-..."

Run with:
    python examples/01_getting_started/first_model_call.py
"""

from __future__ import annotations

from ember.api import models


def demonstrate_discovery() -> None:
    """Show available models and providers."""
    print("Model Discovery")
    print("-" * 40)

    # List available providers
    providers = models.providers()
    print(f"Available providers: {', '.join(providers)}")

    # List some models (limit for display)
    available = models.list()[:10]
    print(f"Sample models: {', '.join(available)}")
    print()


def demonstrate_model_call() -> None:
    """Make a simple model call."""
    print("Making a Model Call")
    print("-" * 40)

    # Simple string response
    prompt = "What is 2 + 2? Reply with just the number."
    print(f"Prompt: {prompt}")

    try:
        response = models("gpt-4o-mini", prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("To fix this, configure your API key:")
        print("  ember configure set providers.openai.api_key YOUR_KEY")
        return

    print()

    # Structured response with metadata
    print("Getting Response Metadata")
    print("-" * 40)

    response_obj = models.response("gpt-4o-mini", "Write a haiku about coding.")
    print(f"Text: {response_obj.text}")
    print(f"Model: {response_obj.model_id}")
    print(f"Usage: {response_obj.usage}")


def main() -> None:
    """Run the first model call example."""
    print("First Model Call with Ember")
    print("=" * 40)
    print()

    demonstrate_discovery()
    demonstrate_model_call()

    print()
    print("Next: Try examples/01_getting_started/model_comparison.py")


if __name__ == "__main__":
    main()
