"""Register Models Directly

This script registers models directly to the registry without using environment variables.
It demonstrates the new simplified API for model registration.

To run:
    uv run python src/ember/examples/models/register_models_directly.py

    # Or if in the virtual env
    python src/ember/examples/models/register_models_directly.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key for registering OpenAI models
    ANTHROPIC_API_KEY (optional): Your Anthropic API key for registering Anthropic models

Note: If your env variables are not set, you can also edit this file to add your API keys directly.
"""

import logging
from typing import List

from prettytable import PrettyTable

from ember.api import models
from ember.api.models import ModelCost, ModelInfo, ModelRegistry, RateLimit

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Use these functions to register models with your API keys
def register_openai_models(api_key: str, registry: ModelRegistry) -> List[str]:
    """Register OpenAI models with the registry.

    Args:
        api_key: OpenAI API key
        registry: ModelRegistry instance

    Returns:
        List of registered model IDs
    """
    # Create the models
    model_infos = [
        ModelInfo(
            id="openai:gpt-4o",
            name="GPT-4o",
            context_window=128000,
            cost=ModelCost(
                input_cost_per_thousand=0.005,  # $0.005 per 1K input tokens
                output_cost_per_thousand=0.015,  # $0.015 per 1K output tokens
            ),
            rate_limit=RateLimit(tokens_per_minute=10000000, requests_per_minute=1500),
            provider={
                "name": "OpenAI",
                "default_api_key": api_key,
                "base_url": "https://api.openai.com/v1",
            },
        ),
        ModelInfo(
            id="openai:gpt-4o-mini",
            name="GPT-4o Mini",
            context_window=128000,
            cost=ModelCost(
                input_cost_per_thousand=0.00015,  # $0.00015 per 1K input tokens
                output_cost_per_thousand=0.0006,  # $0.0006 per 1K output tokens
            ),
            rate_limit=RateLimit(tokens_per_minute=10000000, requests_per_minute=1500),
            provider={
                "name": "OpenAI",
                "default_api_key": api_key,
                "base_url": "https://api.openai.com/v1",
            },
        ),
    ]

    # Register the models
    registered_ids = []
    for model_info in model_infos:
        if not registry.is_registered(model_info.id):
            registry.register_model(model_info=model_info)
            logger.info(f"Registered model: {model_info.id}")
        else:
            logger.info(
                f"Model {model_info.id} already registered ✅ - using existing registration"
            )
        registered_ids.append(model_info.id)

    return registered_ids


def register_anthropic_models(api_key: str, registry: ModelRegistry) -> List[str]:
    """Register Anthropic models with the registry.

    Args:
        api_key: Anthropic API key
        registry: ModelRegistry instance

    Returns:
        List of registered model IDs
    """
    # Create the models
    model_infos = [
        ModelInfo(
            id="anthropic:claude-3-5-sonnet",
            name="Claude 3.5 Sonnet",
            context_window=200000,
            cost=ModelCost(
                input_cost_per_thousand=0.003,  # $0.003 per 1K input tokens
                output_cost_per_thousand=0.015,  # $0.015 per 1K output tokens
            ),
            rate_limit=RateLimit(tokens_per_minute=5000000, requests_per_minute=1000),
            provider={
                "name": "Anthropic",
                "default_api_key": api_key,
                "base_url": "https://api.anthropic.com/v1",
            },
        ),
        ModelInfo(
            id="anthropic:claude-3-opus",
            name="Claude 3 Opus",
            context_window=200000,
            cost=ModelCost(
                input_cost_per_thousand=0.015,  # $0.015 per 1K input tokens
                output_cost_per_thousand=0.075,  # $0.075 per 1K output tokens
            ),
            rate_limit=RateLimit(tokens_per_minute=5000000, requests_per_minute=1000),
            provider={
                "name": "Anthropic",
                "default_api_key": api_key,
                "base_url": "https://api.anthropic.com/v1",
            },
        ),
    ]

    # Register the models
    registered_ids = []
    for model_info in model_infos:
        if not registry.is_registered(model_info.id):
            registry.register_model(model_info=model_info)
            logger.info(f"Registered model: {model_info.id}")
        else:
            logger.info(
                f"Model {model_info.id} already registered ✅ - using existing registration"
            )
        registered_ids.append(model_info.id)

    return registered_ids


def check_models(model_ids: List[str], registry: ModelRegistry) -> None:
    """Check if specific models are available in the registry.

    Args:
        model_ids: List of model IDs to check
        registry: ModelRegistry instance
    """
    # Create a table for displaying model information
    table = PrettyTable()
    table.field_names = [
        "Model ID",
        "Status",
        "Provider",
        "Context Window",
        "Input Cost",
        "Output Cost",
    ]
    table.align = "l"

    for model_id in model_ids:
        exists = registry.is_registered(model_id)
        if exists:
            try:
                info = registry.get_model_info(model_id)
                provider_name = (
                    info.provider.name if hasattr(info, "provider") else "Unknown"
                )
                context_window = getattr(info, "context_window", "N/A")

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
                    [
                        model_id,
                        "✅ Available",
                        provider_name,
                        context_window,
                        input_cost,
                        output_cost,
                    ]
                )

                logger.info(f"Model '{model_id}' is available and initialized")
            except Exception as e:
                table.add_row(
                    [model_id, "⚠️ Error", "Error", "Error", "Error", "Error"]
                )
                logger.warning(f"Error getting model info for '{model_id}': {e}")
        else:
            table.add_row([model_id, "❌ Not Found", "N/A", "N/A", "N/A", "N/A"])
            logger.warning(f"Model '{model_id}' is not available")

    print("\nModel Availability:")
    print(table)


def main():
    """Run the direct model registration example."""
    print("\n=== Direct Model Registration Example ===\n")

    # Initialize the model registry
    registry = models.initialize_registry(auto_discover=True)

    # Enter your API keys here to register the models
    # If you don't want to hardcode them, you would usually get them from environment variables
    # Get keys from environment variables if available
    import os

    openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")

    # Get list of models before registration
    print("Models before registration:")
    all_models = registry.list_models()
    print(f"Found {len(all_models)} models\n")

    # Register the models (uncomment these lines and add your API keys)
    registered_models = []

    if openai_api_key != "YOUR_OPENAI_API_KEY":
        openai_models = register_openai_models(openai_api_key, registry)
        registered_models.extend(openai_models)
    else:
        print("Skipping OpenAI model registration (no API key provided)")

    if anthropic_api_key != "YOUR_ANTHROPIC_API_KEY":
        anthropic_models = register_anthropic_models(anthropic_api_key, registry)
        registered_models.extend(anthropic_models)
    else:
        print("Skipping Anthropic model registration (no API key provided)")

    # Check which models are available
    check_models(
        [
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-5-sonnet",
            "anthropic:claude-3-opus",
        ],
        registry,
    )

    # Example of using a registered model
    print("\nTo use a registered model:")
    print('response = models("openai:gpt-4o", "What is the capital of France?")')
    print("print(response.text)")

    print("\nExample completed! To use models, add your API keys to the script.")


if __name__ == "__main__":
    main()
