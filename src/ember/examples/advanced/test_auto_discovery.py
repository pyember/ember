"""Test the automatic model discovery functionality.

This script tests the automatic model discovery functionality in the model registry.
It's a simple test script that directly uses the registry and discovery service.

To run:
    uv run python src/ember/examples/advanced/test_auto_discovery.py
"""

import logging
import os

# Import correctly from ember packages
from ember.core.registry.model.initialization import initialize_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_auto_discovery():
    """Test the automatic model discovery functionality."""
    # First, initialize the registry with auto_discover=False
    logger.info("Initializing registry with auto_discover=False")
    registry_without_discovery = initialize_registry(auto_discover=False)

    # Check if models were discovered
    model_ids_1 = registry_without_discovery.list_models()
    if model_ids_1:
        logger.info(
            f"Registry without auto-discovery found {len(model_ids_1)} models: {model_ids_1}"
        )
    else:
        logger.info("Registry without auto-discovery has no models as expected")

    # Next, initialize with force_discovery=True
    logger.info("Initializing registry with force_discovery=True")
    registry_with_force = initialize_registry(auto_discover=False, force_discovery=True)

    # Check if models were discovered
    model_ids_2 = registry_with_force.list_models()
    if model_ids_2:
        logger.info(
            f"Registry with force_discovery found {len(model_ids_2)} models: {model_ids_2}"
        )
    else:
        logger.warning("Registry with force_discovery did not find any models")

    # Finally, initialize with auto_discover=True
    logger.info("Initializing registry with auto_discover=True")
    registry_with_auto = initialize_registry(auto_discover=True)

    # Check if models were discovered
    model_ids_3 = registry_with_auto.list_models()
    if model_ids_3:
        logger.info(
            f"Registry with auto_discover found {len(model_ids_3)} models: {model_ids_3}"
        )
    else:
        logger.warning("Registry with auto_discover did not find any models")

    # Select the registry with the most models for detailed display
    if len(model_ids_3) >= len(model_ids_2) and len(model_ids_3) >= len(model_ids_1):
        best_registry = registry_with_auto
    elif len(model_ids_2) >= len(model_ids_1):
        best_registry = registry_with_force
    else:
        best_registry = registry_without_discovery

    # Print all models in the best registry
    logger.info("All models in best registry:")
    for model_id in best_registry.list_models():
        model_info = best_registry.get_model_info(model_id)
        if model_info:
            provider_name = (
                model_info.provider.name
                if hasattr(model_info.provider, "name")
                else "Unknown"
            )
            context_window = (
                model_info.context_window
                if hasattr(model_info, "context_window")
                else "N/A"
            )
            cost = model_info.cost if hasattr(model_info, "cost") else None

            if cost:
                i_cost = (
                    f"${cost.input_cost_per_thousand:.5f}/1K"
                    if hasattr(cost, "input_cost_per_thousand")
                    else "N/A"
                )
                o_cost = (
                    f"${cost.output_cost_per_thousand:.5f}/1K"
                    if hasattr(cost, "output_cost_per_thousand")
                    else "N/A"
                )
                cost_str = f"(In: {i_cost}, Out: {o_cost})"
            else:
                cost_str = "(Cost: N/A)"

            logger.info(
                f"  - {model_id}: Provider={provider_name}, Context={context_window} {cost_str}"
            )

    return best_registry


if __name__ == "__main__":
    logger.info("Testing automatic model discovery...")

    # Check which API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }

    for key, value in api_keys.items():
        if value:
            logger.info(f"{key} is set - length: {len(value[:3])}...")
        else:
            logger.warning(f"{key} is NOT set")

    # Run the test
    test_auto_discovery()
