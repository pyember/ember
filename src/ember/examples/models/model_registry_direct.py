"""
Direct Model Registry Example with environment variable API keys.

This example demonstrates how to directly use the model registry with API keys
from environment variables.

To run:
    export OPENAI_API_KEY="your-openai-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
    uv run python src/ember/examples/models/model_registry_direct.py

    # Or if in an activated virtual environment
    python src/ember/examples/models/model_registry_direct.py
"""

import logging
import os

# Import the registry components
from ember import initialize_ember
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run a direct model registry example."""
    print("\n=== Direct Model Registry Example ===\n")

    # Initialize the registry with no auto-discovery
    registry = initialize_ember(auto_discover=False, initialize_context=False)

    # Get API keys from environment variables
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key:
        logger.warning(
            "OPENAI_API_KEY environment variable not set. OpenAI example will fail."
        )

    if not anthropic_key:
        logger.warning(
            "ANTHROPIC_API_KEY environment variable not set. Anthropic example will fail."
        )

    # Create provider info objects
    openai_provider = ProviderInfo(
        name="OpenAI", default_api_key=openai_key, base_url="https://api.openai.com/v1"
    )

    anthropic_provider = ProviderInfo(
        name="Anthropic",
        default_api_key=anthropic_key,
        base_url="https://api.anthropic.com/v1",
    )

    # Create and register model info
    gpt4o_model = ModelInfo(id="openai:gpt-4o", name="gpt-4o", provider=openai_provider)

    claude_model = ModelInfo(
        id="anthropic:claude-3-opus",
        name="claude-3-opus",
        provider=anthropic_provider,
    )

    # Register the models
    registry.register_model(model_info=gpt4o_model)
    registry.register_model(model_info=claude_model)

    # Create a model service
    usage_service = UsageService()
    model_service = ModelService(registry=registry, usage_service=usage_service)

    # Try to use the models
    if openai_key:
        try:
            print("Trying OpenAI GPT-4o:")
            openai_response = model_service(
                "openai:gpt-4o", "What is the capital of France?"
            )
            print(f"Response: {openai_response.data}")
        except Exception as e:
            print(f"Error with OpenAI: {e}")
    else:
        print("Skipping OpenAI example because OPENAI_API_KEY is not set.")

    if anthropic_key:
        try:
            print("\nTrying Anthropic Claude:")
            anthropic_response = model_service(
                "anthropic:claude-3-opus", "What is the capital of Italy?"
            )
            print(f"Response: {anthropic_response.data}")
        except Exception as e:
            print(f"Error with Anthropic: {e}")
    else:
        print("\nSkipping Anthropic example because ANTHROPIC_API_KEY is not set.")

    print("\nExample completed!")


if __name__ == "__main__":
    main()
