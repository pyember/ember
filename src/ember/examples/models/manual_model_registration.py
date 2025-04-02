"""Manual Model Registration Example

This example demonstrates how to manually register models with the ModelRegistry
using the new simplified API.

IMPORTANT: Model pricing and context window information must be manually configured!
When models are discovered via API, they DO NOT include pricing or context window 
information automatically. You must:

1. Add this information in your config.yaml file in the project root:
   ```yaml
   model_registry:
     providers:
       openai:
         models:
           - id: "gpt-4o"
             name: "GPT-4o"
             context_window: 128000  # <-- Add this for context window
             cost:
               input_cost_per_thousand: 5.0
               output_cost_per_thousand: 15.0
   ```

2. Or register models with complete information in code using the ModelInfo class
   as shown in the register_openai_models() and register_anthropic_models() functions below.

See model pricing and specifications:
https://docs.anthropic.com/en/docs/about-claude/models/all-models
https://openai.com/api/pricing/

To run:
    uv run python src/ember/examples/models/manual_model_registration.py

    # Or if in an activated virtual environment
    python src/ember/examples/models/manual_model_registration.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key for registering OpenAI models
    ANTHROPIC_API_KEY (optional): Your Anthropic API key for registering Anthropic models
"""

import logging
import os
from typing import List

from prettytable import PrettyTable

from ember.api import models
from ember.api.models import ModelCost, ModelInfo, ProviderInfo, RateLimit

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def register_models() -> List[str]:
    """Register models manually with the registry.

    Returns:
        List of registered model IDs
    """
    # Get API keys from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Get the registry
    registry = models.get_registry()

    registered_models = []

    if openai_api_key:
        # Register OpenAI models
        # Register GPT-4o
        gpt4o_info = ModelInfo(
            id="openai:gpt-4o",
            name="GPT-4o",
            context_window=128000,
            cost=ModelCost(
                input_cost_per_thousand=0.005,  # $0.005 per 1K input tokens
                output_cost_per_thousand=0.015,  # $0.015 per 1K output tokens
            ),
            rate_limit=RateLimit(
                tokens_per_minute=10000000,  # Up to 10 million tokens per minute
                requests_per_minute=1500,  # Tier 5 rate limit
            ),
            provider=ProviderInfo(
                name="OpenAI",
                default_api_key=openai_api_key,
                base_url="https://api.openai.com/v1",
            ),
        )
        if not registry.is_registered("openai:gpt-4o"):
            try:
                registry.register_model(model_info=gpt4o_info)
                logger.info("Registered model: openai:gpt-4o")
            except ValueError as e:
                logger.info(f"Model registration error: {e}")
        else:
            logger.info("Model openai:gpt-4o already registered, skipping")

        registered_models.append("openai:gpt-4o")

        # Register GPT-4o-mini
        gpt4o_mini_info = ModelInfo(
            id="openai:gpt-4o-mini",
            name="GPT-4o Mini",
            context_window=128000,
            cost=ModelCost(
                input_cost_per_thousand=0.00015,  # $0.00015 per 1K input tokens
                output_cost_per_thousand=0.0006,  # $0.0006 per 1K output tokens
            ),
            rate_limit=RateLimit(tokens_per_minute=10000000, requests_per_minute=1500),
            provider=ProviderInfo(
                name="OpenAI",
                default_api_key=openai_api_key,
                base_url="https://api.openai.com/v1",
            ),
        )
        if not registry.is_registered("openai:gpt-4o-mini"):
            try:
                registry.register_model(model_info=gpt4o_mini_info)
                logger.info("Registered model: openai:gpt-4o-mini")
            except ValueError as e:
                logger.info(f"Model registration error: {e}")
        else:
            logger.info("Model openai:gpt-4o-mini already registered, skipping")

        registered_models.append("openai:gpt-4o-mini")
    else:
        logger.warning("OPENAI_API_KEY not found in environment variables")

    if anthropic_api_key:
        # Register Anthropic models
        # Register Claude 3.5 Sonnet
        claude_sonnet_info = ModelInfo(
            id="anthropic:claude-3-5-sonnet",
            name="Claude 3.5 Sonnet",
            context_window=200000,
            cost=ModelCost(
                input_cost_per_thousand=0.003,  # $0.003 per 1K input tokens
                output_cost_per_thousand=0.015,  # $0.015 per 1K output tokens
            ),
            rate_limit=RateLimit(tokens_per_minute=5000000, requests_per_minute=1000),
            provider=ProviderInfo(
                name="Anthropic",
                default_api_key=anthropic_api_key,
                base_url="https://api.anthropic.com/v1",
            ),
        )
        if not registry.is_registered("anthropic:claude-3-5-sonnet"):
            try:
                registry.register_model(model_info=claude_sonnet_info)
                logger.info("Registered model: anthropic:claude-3-5-sonnet")
            except ValueError as e:
                logger.info(f"Model registration error: {e}")
        else:
            logger.info(
                "Model anthropic:claude-3-5-sonnet already registered, skipping"
            )

        registered_models.append("anthropic:claude-3-5-sonnet")
    else:
        logger.warning("ANTHROPIC_API_KEY not found in environment variables")

    # Log all registered models
    logger.info(f"Registered models: {registered_models}")

    return registered_models


def display_models():
    """Display information about the registered models in a formatted table."""
    registry = models.get_registry()
    model_ids = registry.list_models()

    if not model_ids:
        print("No models found in the registry.")
        return

    # Create a table for displaying model information
    table = PrettyTable()
    table.field_names = [
        "Model ID",
        "Provider",
        "Context Window",
        "Input Cost",
        "Output Cost",
    ]
    table.align = "l"

    # Group by provider
    providers = {}
    for model_id in model_ids:
        if ":" in model_id:
            provider, name = model_id.split(":", 1)
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model_id)
        else:
            # Handle models without provider prefix
            if "other" not in providers:
                providers["other"] = []
            providers["other"].append(model_id)

    # Add models to table
    for provider, ids in sorted(providers.items()):
        for model_id in sorted(ids):
            try:
                info = registry.get_model_info(model_id)
                provider_name = provider
                if hasattr(info, "provider"):
                    if isinstance(info.provider, dict):
                        provider_name = info.provider.get("name", provider)
                    elif hasattr(info.provider, "name"):
                        provider_name = info.provider.name
                context_window = (
                    info.context_window if hasattr(info, "context_window") else "N/A"
                )

                input_cost = (
                    f"${info.cost.input_cost_per_thousand:.4f}/1K"
                    if hasattr(info, "cost") and info.cost
                    else "N/A"
                )
                output_cost = (
                    f"${info.cost.output_cost_per_thousand:.4f}/1K"
                    if hasattr(info, "cost") and info.cost
                    else "N/A"
                )

                table.add_row(
                    [model_id, provider_name, context_window, input_cost, output_cost]
                )
            except Exception as e:
                table.add_row([model_id, f"Error: {str(e)[:20]}...", "", "", ""])

    print("\nRegistered Models:")
    print(table)


def main():
    """Run the manual model registration example."""
    print("\n=== Manual Model Registration Example ===\n")

    # Display models before registration
    print("Models before registration:")
    display_models()

    # Register models
    registered_model_ids = register_models()

    # Display updated models
    print("\nModels after registration:")
    display_models()

    # Try to use one of the registered models
    if "openai:gpt-4o" in registered_model_ids:
        try:
            print("\n=== Using a registered model ===")
            model_service = models.get_model_service()
            response = model_service.invoke_model(
                "openai:gpt-4o", "What is the capital of France?"
            )
            print("Query: What is the capital of France?")
            print(f"Response: {response.data[:150]}...")
            print(f"Usage stats: {response.usage}")
        except Exception as e:
            logger.error(f"Error using model: {e}")

    # Example code for using a model
    print("\nTo use a registered model with the API:")
    print("from ember.api import models")
    print("# Using the model service")
    print("model_service = models.get_model_service()")
    print(
        'response = model_service.invoke_model("openai:gpt-4o", "What is the capital of France?")'
    )
    print("print(response.data)")
    print("\n# Or using the provider namespace (preferred)")
    print('response = models.openai.gpt4o("What is the capital of France?")')
    print("print(response.data)")

    print("\nExample completed!")


if __name__ == "__main__":
    main()
