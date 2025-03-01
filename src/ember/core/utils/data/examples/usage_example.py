"""
Usage:
    python -m ember.core.utils.data.examples.usage_example

This example demonstrates the high-level dataset loading and preparation pipeline.
It shows how to retrieve processed dataset entries using two different methods:
  1. Loading MMLU dataset entries using a configuration dictionary.
  2. Loading HaluEval dataset entries using a configuration instance.
"""

import logging
from typing import Any, List

# Import for the high-level one-liner API function.
from ember.core.utils.data import load_dataset_entries

# Import for configuration classes.
from ember.core.utils.data.datasets_registry.halueval import HaluEvalConfig
from ember.core.utils.data.datasets_registry.mmlu import MMLUConfig


def main() -> None:
    """Demonstrate usage of the high-level dataset loading pipeline.

    Raises:
        Exception: Propagates any errors encountered during the dataset loading process.
    """
    # Configure logging at the INFO level.
    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)

    # Example 1: Load and prepare MMLU dataset entries using a configuration dictionary.
    try:
        logger.info("Loading MMLU dataset entries with configuration dictionary.")
        mmlu_entries: List[Any] = load_dataset_entries(
            dataset_name="mmlu",
            config=MMLUConfig(
                config_name="abstract_algebra",
                split="dev",
            ),
            num_samples=5,
        )
        logger.info("Successfully loaded %d entries for MMLU.", len(mmlu_entries))
        for idx, entry in enumerate(mmlu_entries, start=1):
            logger.info("MMLU Entry #%d:\n%s", idx, entry.model_dump_json(indent=2))
    except Exception as error:
        logger.exception("Error loading MMLU dataset entries: %s", error)

    # Example 2: Load and prepare HaluEval dataset entries using a configuration instance.
    try:
        logger.info("Loading HaluEval dataset entries with configuration instance.")
        halu_config: HaluEvalConfig = (
            HaluEvalConfig()
        )  # Defaults: config_name="qa", split="data"
        halu_entries: List[Any] = load_dataset_entries(
            dataset_name="halueval",
            config=halu_config,
            num_samples=3,
        )
        logger.info("Successfully loaded %d entries for HaluEval.", len(halu_entries))
        for idx, entry in enumerate(halu_entries, start=1):
            logger.info("HaluEval Entry #%d:\n%s", idx, entry.model_dump_json(indent=2))
    except Exception as error:
        logger.exception("Error loading HaluEval dataset entries: %s", error)


if __name__ == "__main__":
    main()
