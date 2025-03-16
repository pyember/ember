"""Example usage of the Ember data API.

This example demonstrates the intuitive API for working with datasets in Ember,
including basic loading, advanced configuration, and custom dataset registration.

To run:
    poetry run python src/ember/examples/data/data_api_example.py
"""

from typing import Dict, List, Optional

from ember.api import DatasetBuilder, DatasetEntry, TaskType, datasets, register


def basic_usage() -> None:
    """Demonstrate basic dataset usage with direct loading."""
    # Direct loading by dataset name
    mmlu_dataset = datasets("mmlu")
    print(f"Loaded MMLU dataset with {len(mmlu_dataset)} entries")

    # Print the first two entries for demonstration
    for i, entry in enumerate(mmlu_dataset):
        if i >= 2:  # Just show a couple of examples
            break

        print(f"\nEntry {i + 1}:")

        # Access content with safe type handling
        question = entry.content.get("question", "No question available")
        choices = entry.content.get("choices", [])
        answer_idx = entry.content.get("answer", -1)

        print(f"Question: {question}")
        print("Choices:")
        for j, choice in enumerate(choices):
            print(f"  {j}. {choice}")

        # Safe access to answer
        if 0 <= answer_idx < len(choices):
            print(f"Answer: {choices[answer_idx]} (index {answer_idx})")
        else:
            print("Answer: Not available")

        # Print metadata if available
        if entry.metadata:
            print("Metadata:")
            for key, value in entry.metadata.items():
                print(f"  {key}: {value}")


def advanced_usage() -> None:
    """Demonstrate advanced dataset usage with the builder pattern."""
    # Use the builder pattern for configuration
    dataset = (
        DatasetBuilder()
        .split("test")  # Specify the dataset split
        .sample(1)  # Take only one sample
        .seed(42)  # Set random seed for reproducibility
        .build("mmlu")  # Build the configured dataset
    )

    print(f"\nLoaded 1 sample from MMLU test split:")
    if len(dataset) > 0:
        entry = dataset[0]

        # Safe access to content with fallbacks
        question = entry.content.get("question", "No question available")
        choices = entry.content.get("choices", [])
        answer_idx = entry.content.get("answer", -1)

        print(f"Question: {question}")
        print("Choices:")
        for j, choice in enumerate(choices):
            print(f"  {j}. {choice}")

        # Safe access to answer
        if 0 <= answer_idx < len(choices):
            print(f"Answer: {choices[answer_idx]} (index {answer_idx})")
        else:
            print("Answer: Not available")

        # Get domain from metadata with fallback
        print(f"Domain: {entry.metadata.get('domain', 'unknown')}")


def cli_usage() -> None:
    """Demonstrate how to use the CLI for dataset operations."""
    print("\nCLI Usage:")
    print("To list all datasets:")
    print("  ember dataset list")
    print("\nTo show dataset info:")
    print("  ember dataset info mmlu")
    print("\nTo explore a dataset:")
    print("  ember dataset explore mmlu --count 5")
    print("\nTo export a dataset:")
    print("  ember dataset export mmlu --split test --count 10 --output mmlu.json")


def custom_dataset() -> None:
    """Demonstrate how to create and register a custom dataset."""

    # Register a custom dataset (for demonstration only)
    @register("custom_qa", source="custom/qa", task_type=TaskType.SHORT_ANSWER)
    class CustomQADataset:
        """A custom QA dataset for demonstration purposes."""

        def __init__(self) -> None:
            """Initialize the dataset."""
            pass

        def load(self, config: Optional[Dict] = None) -> List[DatasetEntry]:
            """Load the custom dataset.

            Args:
                config: Optional configuration for loading

            Returns:
                List of dataset entries
            """
            # In a real implementation, this would load from a source
            return [
                DatasetEntry(
                    query="What is the capital of France?",
                    metadata={"category": "geography", "answer": "Paris"},
                ),
                DatasetEntry(
                    query="What is the largest mammal?",
                    metadata={"category": "biology", "answer": "Blue whale"},
                ),
            ]

    # Now we can use our custom dataset
    try:
        custom_data = datasets("custom_qa")
        print(f"\nLoaded custom dataset with {len(custom_data)} entries")

        for entry in custom_data:
            # Safely access fields with appropriate error handling
            print(f"\nQuestion: {entry.query}")
            print(f"Answer: {entry.metadata.get('answer', 'Unknown')}")
            print(f"Category: {entry.metadata.get('category', 'Uncategorized')}")
    except Exception as e:
        print(f"Error loading custom dataset: {str(e)}")


def main() -> None:
    """Run the data API example with all demonstrations."""
    print("Ember Data API Example\n" + "=" * 24 + "\n")

    try:
        # Basic usage
        basic_usage()

        # Advanced usage
        advanced_usage()

        # CLI usage info
        cli_usage()

        # Custom dataset
        custom_dataset()

        print("\nFor more information, see the documentation in the project repository")
    except Exception as e:
        print(f"Error running example: {str(e)}")


if __name__ == "__main__":
    main()
