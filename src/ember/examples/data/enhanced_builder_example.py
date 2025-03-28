"""Example demonstrating the enhanced DatasetBuilder pattern in Ember.

This example shows how to use the DatasetBuilder to load, transform, and work with
datasets using a fluent interface.

To run:
    uv run python src/ember/examples/data/enhanced_builder_example.py
"""

from typing import Any, Dict, List

from ember.api.data import DatasetBuilder, list_available_datasets


def uppercase_transformer(item: Dict[str, Any]) -> Dict[str, Any]:
    """Transform dataset items by uppercasing text fields.

    Args:
        item: Dataset item to transform

    Returns:
        Transformed item with uppercase text fields
    """
    result = item.copy()

    # Convert question to uppercase if present
    if "question" in result:
        result["question"] = result["question"].upper()

    # Convert choices to uppercase if present
    if "choices" in result:
        if isinstance(result["choices"], dict):
            # Handle dictionary-style choices
            result["choices"] = {
                k: v.upper() if isinstance(v, str) else v
                for k, v in result["choices"].items()
            }
        elif isinstance(result["choices"], list):
            # Handle list-style choices (MMLU format)
            result["choices"] = [
                c.upper() if isinstance(c, str) else c for c in result["choices"]
            ]

    return result


def prompt_formatter(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format dataset items into LLM-ready prompts.

    Args:
        item: Dataset item to format

    Returns:
        Item with added formatted prompt
    """
    result = item.copy()

    if "question" in result and "choices" in result:
        choices_text = ""

        if isinstance(result["choices"], dict):
            # Handle dictionary-style choices
            for key, value in result["choices"].items():
                choices_text += f"{key}. {value}\n"
        elif isinstance(result["choices"], list):
            # Handle list-style choices (MMLU format)
            options = ["A", "B", "C", "D"]
            for i, choice in enumerate(result["choices"]):
                if i < len(options):
                    choices_text += f"{options[i]}. {choice}\n"

        result["formatted_prompt"] = (
            f"Question: {result['question']}\n\n"
            f"Answer choices:\n{choices_text}\n"
            f"Please select the correct answer choice."
        )

    return result


def main() -> None:
    """Run the enhanced DatasetBuilder example."""
    print("Ember Enhanced DatasetBuilder Example")
    print("=====================================")

    # List available datasets
    print("\nAvailable datasets in registry:")
    datasets = list_available_datasets()
    for dataset in datasets:
        print(f"  • {dataset}")

    # Use the enhanced builder pattern
    print("\nLoading dataset with builder pattern...")

    try:
        # Chain multiple configuration and transformation steps
        dataset = (
            DatasetBuilder()
            .from_registry("mmlu")  # Use a registered dataset
            .subset("high_school_mathematics")  # Select a specific subset
            .split("test")  # Choose the test split
            .sample(3)  # Random sample of 3 items
            .transform(uppercase_transformer)  # Transform to uppercase
            .transform(prompt_formatter)  # Format as prompts
            .build()
        )

        # Display the loaded dataset
        print(f"\nLoaded dataset with {len(dataset)} entries")

        for i, entry in enumerate(dataset):
            print(f"\nEntry {i+1}:")

            # Display query from entry
            print(f"Query: {entry.query}")

            # Display formatted prompt if available
            if hasattr(entry, "metadata") and entry.metadata:
                if "formatted_prompt" in entry.metadata:
                    print(f"\n{'-'*40}")
                    print(f"{entry.metadata['formatted_prompt']}")
                    print(f"{'-'*40}")

                # Display other metadata
                print("\nMetadata:")
                for key, value in entry.metadata.items():
                    if key != "formatted_prompt":
                        print(f"  {key}: {value}")

            # Display choices if available
            if hasattr(entry, "choices") and entry.choices:
                print("\nChoices:")
                for key, value in entry.choices.items():
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
