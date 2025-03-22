"""
Direct Model Registry Example with manually specified API keys.

This example demonstrates how to directly use the model registry with manually
specified API keys instead of using environment variables.

To run:
    uv run python src/ember/examples/models/model_registry_direct.py

    # Or if in an activated virtual environment
    python src/ember/examples/models/model_registry_direct.py

Note: You need to edit this file to replace the placeholder API keys with your actual keys.
"""

import logging
from typing import Dict, List, Optional

# Import the registry components
from ember import initialize_ember
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
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

    # Manually specify API keys (replace with your actual keys)
    # WARNING: Do not commit actual API keys to source control. This is just for demonstration.
    openai_key = "YOUR_OPENAI_API_KEY_HERE"
    anthropic_key = "YOUR_ANTHROPIC_API_KEY_HERE"

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
    gpt4o_model = ModelInfo(
        id="openai:gpt-4o", model_name="gpt-4o", provider=openai_provider
    )

    claude_model = ModelInfo(
        id="anthropic:claude-3-opus",
        model_name="claude-3-opus",
        provider=anthropic_provider,
    )

    # Register the models
    registry.register_model(model_info=gpt4o_model)
    registry.register_model(model_info=claude_model)

    # Create a model service
    usage_service = UsageService()
    model_service = ModelService(registry=registry, usage_service=usage_service)

    # Try to use the models
    try:
        print("Trying OpenAI GPT-4o:")
        openai_response = model_service(
            "openai:gpt-4o", "What is the capital of France?"
        )
        print(f"Response: {openai_response.data}")
    except Exception as e:
        print(f"Error with OpenAI: {e}")

    try:
        print("\nTrying Anthropic Claude:")
        anthropic_response = model_service(
            "anthropic:claude-3-opus", "What is the capital of Italy?"
        )
        print(f"Response: {anthropic_response.data}")
    except Exception as e:
        print(f"Error with Anthropic: {e}")

    print("\nExample completed!")


if __name__ == "__main__":
    main()
