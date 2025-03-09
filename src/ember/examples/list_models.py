"""Ember Model Discovery Example

This script demonstrates how to list available models in the Ember registry
using the simplified API. The script shows how to check for model availability
and retrieve model information.

To run:
    poetry run python src/ember/examples/list_models.py
"""

import os
import logging
from typing import List, Dict, Any
from prettytable import PrettyTable

from ember.api import models

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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
    model_ids = models.registry.list_models()
    
    # Create a table for display
    table = PrettyTable()
    table.field_names = ["Provider", "Model ID", "Context Window", "Input Cost", "Output Cost"]
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
    print(f"\nFound {len(model_ids)} models across {len(providers)} providers")
    
    # Add models to table
    for provider, ids in sorted(providers.items()):
        for model_id in sorted(ids):
            try:
                info = models.registry.get_model_info(model_id)
                table.add_row([
                    provider,
                    model_id,
                    f"{info.context_window if hasattr(info, 'context_window') else 'N/A'}",
                    f"${info.cost.input_cost_per_thousand:.4f}" if hasattr(info, 'cost') and info.cost else "N/A",
                    f"${info.cost.output_cost_per_thousand:.4f}" if hasattr(info, 'cost') and info.cost else "N/A"
                ])
            except Exception as e:
                table.add_row([provider, model_id, "Error", "Error", "Error"])
    
    print(table)

def check_specific_models(model_ids: List[str]):
    """Check if specific models are available in the registry.
    
    Args:
        model_ids: List of model IDs to check
    """
    print("\nChecking specific models:")
    for model_id in model_ids:
        exists = models.registry.model_exists(model_id)
        if exists:
            info = models.registry.get_model_info(model_id)
            logger.info(f"✅ Model '{model_id}' is available")
            print(f"   - Provider: {info.provider['name'] if 'name' in info.provider else 'Unknown'}")
            if hasattr(info, 'cost') and info.cost:
                print(f"   - Input cost: ${info.cost.input_cost_per_thousand:.4f} per 1K tokens")
                print(f"   - Output cost: ${info.cost.output_cost_per_thousand:.4f} per 1K tokens")
        else:
            logger.warning(f"❌ Model '{model_id}' is not available")

def main():
    """Run the model discovery example."""
    print("\n=== Ember Model Discovery Example ===\n")
    
    # Check if API keys are set
    check_api_keys()
    
    # List all available models
    list_available_models()
    
    # Check specific models
    check_specific_models([
        "openai:gpt-4o", 
        "openai:gpt-4o-mini", 
        "anthropic:claude-3.5-sonnet",
        "anthropic:claude-3-opus"
    ])
    
    # Example of the simpler usage pattern
    print("\nUsing simpler direct model identification:")
    print("To check if a model exists: models.registry.model_exists('openai:gpt-4o')")
    print("To get model info: models.registry.get_model_info('openai:gpt-4o')")
    print("To use a model: models.openai.gpt4o('What is the capital of France?')")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()