"""Ember Model Discovery Example

This script demonstrates how to list available models in the Ember registry
using the simplified API. The script shows how to check for model availability
and retrieve model information.

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
    uv run python src/ember/examples/models/list_models.py
"""

import logging
import os
from typing import List

from prettytable import PrettyTable

from ember.api import models

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize the registry with auto_discover=True
# Using the get_registry() function will initialize the registry if needed
logger.info("Initializing registry with auto_discover=True...")
registry = models.get_registry()

# Check if models were discovered during initialization
model_ids = registry.list_models()
if model_ids:
    logger.info(
        f"Successfully discovered {len(model_ids)} models during initialization: {model_ids}"
    )
else:
    logger.info(
        "No models discovered during initialization, attempting manual discovery..."
    )

    # Try explicit discovery
    discovered_models = registry.discover_models()

    if discovered_models:
        logger.info(
            f"Manual discovery found {len(discovered_models)} models: {discovered_models}"
        )
    else:
        logger.info("No models discovered, falling back to manual registration")


def register_openai_models():
    """Register OpenAI models with the registry."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set, skipping OpenAI model registration")
        return []

    # Create the models
    model_infos = [
        models.ModelInfo(
            id="openai:gpt-4o",
            name="GPT-4o",
            context_window=128000,
            cost=models.ModelCost(
                input_cost_per_thousand=0.005, output_cost_per_thousand=0.015
            ),
            rate_limit=models.RateLimit(
                tokens_per_minute=10000000, requests_per_minute=1500
            ),
            provider={
                "name": "OpenAI",
                "default_api_key": openai_key,
                "base_url": "https://api.openai.com/v1",
            },
        ),
        models.ModelInfo(
            id="openai:gpt-4o-mini",
            name="GPT-4o Mini",
            context_window=128000,
            cost=models.ModelCost(
                input_cost_per_thousand=0.00015, output_cost_per_thousand=0.0006
            ),
            rate_limit=models.RateLimit(
                tokens_per_minute=10000000, requests_per_minute=1500
            ),
            provider={
                "name": "OpenAI",
                "default_api_key": openai_key,
                "base_url": "https://api.openai.com/v1",
            },
        ),
    ]

    # Register the models, skipping any that are already registered
    registered_models = []
    for model_info in model_infos:
        if not registry.is_registered(model_info.id):
            try:
                registry.register_model(model_info=model_info)
                logger.info(f"Registered model: {model_info.id}")
                registered_models.append(model_info.id)
            except ValueError:
                logger.info(f"Model {model_info.id} already registered, skipping")
        else:
            logger.info(f"Model {model_info.id} already registered, skipping")

    return registered_models


def register_anthropic_models():
    """Register Anthropic models with the registry."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set, skipping Anthropic model registration"
        )
        return []

    # Create the models
    model_infos = [
        models.ModelInfo(
            id="anthropic:claude-3-5-sonnet",
            name="Claude 3.5 Sonnet",
            context_window=200000,
            cost=models.ModelCost(
                input_cost_per_thousand=0.003, output_cost_per_thousand=0.015
            ),
            rate_limit=models.RateLimit(
                tokens_per_minute=5000000, requests_per_minute=1000
            ),
            provider={
                "name": "Anthropic",
                "default_api_key": anthropic_key,
                "base_url": "https://api.anthropic.com/v1",
            },
        ),
        models.ModelInfo(
            id="anthropic:claude-3-opus",
            name="Claude 3 Opus",
            context_window=200000,
            cost=models.ModelCost(
                input_cost_per_thousand=0.015, output_cost_per_thousand=0.075
            ),
            rate_limit=models.RateLimit(
                tokens_per_minute=5000000, requests_per_minute=1000
            ),
            provider={
                "name": "Anthropic",
                "default_api_key": anthropic_key,
                "base_url": "https://api.anthropic.com/v1",
            },
        ),
    ]

    # Register the models, skipping any that are already registered
    registered_models = []
    for model_info in model_infos:
        if not registry.is_registered(model_info.id):
            try:
                registry.register_model(model_info=model_info)
                logger.info(f"Registered model: {model_info.id}")
                registered_models.append(model_info.id)
            except ValueError:
                logger.info(f"Model {model_info.id} already registered, skipping")
        else:
            logger.info(f"Model {model_info.id} already registered, skipping")

    return registered_models


# Register models manually as a fallback
register_openai_models()
register_anthropic_models()


def check_api_keys():
    """Check if API keys are set in environment variables."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        logger.info("OPENAI_API_KEY is set")
    else:
        logger.warning("OPENAI_API_KEY is not set")

    if anthropic_key:
        logger.info("ANTHROPIC_API_KEY is set")
    else:
        logger.warning("ANTHROPIC_API_KEY is not set")


def list_available_models():
    """List available models in the registry using the new API.

    With the new API, model discovery happens automatically.
    """
    logger.info("Listing available models...")

    # Get all available models
    model_ids = registry.list_models()

    # Create a table for display
    table = PrettyTable()
    table.field_names = [
        "Provider",
        "Model ID",
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

    # Print models by provider
    logger.info(f"Found {len(model_ids)} models across {len(providers)} providers")

    # Add models to table
    for provider, ids in sorted(providers.items()):
        for model_id in sorted(ids):
            try:
                info = registry.get_model_info(model_id)
                table.add_row(
                    [
                        provider,
                        model_id,
                        f"{info.context_window if hasattr(info, 'context_window') else 'N/A'}",
                        (
                            f"${info.cost.input_cost_per_thousand:.4f}"
                            if hasattr(info, "cost") and info.cost
                            else "N/A"
                        ),
                        (
                            f"${info.cost.output_cost_per_thousand:.4f}"
                            if hasattr(info, "cost") and info.cost
                            else "N/A"
                        ),
                    ]
                )
            except Exception:
                table.add_row([provider, model_id, "Error", "Error", "Error"])

    logger.info(f"Model table:\n{table}")


def check_specific_models(model_ids: List[str]):
    """Check if specific models are available in the registry.

    Args:
        model_ids: List of model IDs to check
    """
    logger.info("Checking specific models:")
    for model_id in model_ids:
        exists = registry.is_registered(model_id)
        if exists:
            info = registry.get_model_info(model_id)
            logger.info(f"✅ Model '{model_id}' is available")
            logger.info(f"   - Provider: {info.provider.name if hasattr(info.provider, 'name') else 'Unknown'}")
            if hasattr(info, "cost") and info.cost:
                logger.info(f"   - Input cost: ${info.cost.input_cost_per_thousand:.4f} per 1K tokens")
                logger.info(f"   - Output cost: ${info.cost.output_cost_per_thousand:.4f} per 1K tokens")
        else:
            logger.warning(f"❌ Model '{model_id}' is not available")


def main():
    """Run the model discovery example."""
    logger.info("=== Ember Model Discovery Example ===")

    # Check if API keys are set
    check_api_keys()

    # List all available models
    list_available_models()

    # Check specific models
    check_specific_models(
        [
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-5-sonnet",
            "anthropic:claude-3-opus",
        ]
    )

    # Example of the simpler usage pattern
    logger.info("Using simpler direct model identification:")
    logger.info("To check if a model exists: registry.is_registered('openai:gpt-4o')")
    logger.info("To get model info: registry.get_model_info('openai:gpt-4o')")
    logger.info("To use a model: model_service = models.create_model_service(registry)")
    logger.info("               model_service.invoke_model('openai:gpt-4o', 'What is the capital of France?')")
    
    logger.info("Example completed!")


if __name__ == "__main__":
    main()
