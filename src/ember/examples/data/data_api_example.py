"""Example usage of the Ember data API.

This example demonstrates the intuitive API for working with datasets in Ember,
including basic loading, advanced configuration, and custom dataset registration.

To run:
    poetry run python src/ember/examples/data/data_api_example.py
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from datasets import Dataset, DatasetDict

from ember.api import DatasetBuilder, DatasetEntry, TaskType, register
from ember.core.utils.data.base.loaders import IDatasetLoader
from ember.core.utils.data.base.models import DatasetInfo
from ember.core.utils.data.base.samplers import DatasetSampler
from ember.core.utils.data.base.transformers import NoOpTransformer
from ember.core.utils.data.base.validators import DatasetValidator
from ember.core.utils.data.registry import UNIFIED_REGISTRY, initialize_registry
from ember.core.utils.data.service import DatasetService

# Initialize the registry with predefined datasets
initialize_registry()

# Define a custom dataset class registry to store the class implementations
custom_dataset_registry = {}


def register_mock_dataset() -> None:
    """Register a mock dataset for demonstration purposes."""

    @register(
        "mock_qa",
        source="custom/mock_qa",
        task_type=TaskType.MULTIPLE_CHOICE,
        description="A mock dataset for multiple choice QA examples",
    )
    class MockQADataset:
        """A mock QA dataset for demonstration purposes."""

        def __init__(self) -> None:
            """Initialize the dataset."""
            pass

        def load(self, config: Optional[Dict] = None) -> List[DatasetEntry]:
            """Load the mock dataset.

            Args:
                config: Optional configuration for loading

            Returns:
                List of dataset entries
            """
            # In a real implementation, this would load from a real source
            return [
                DatasetEntry(
                    content={
                        "question": "What is the capital of France?",
                        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
                        "answer": 2,  # Paris (0-indexed)
                    },
                    metadata={"category": "geography", "difficulty": "easy"},
                ),
                DatasetEntry(
                    content={
                        "question": "Which of these is a mammal?",
                        "choices": ["Shark", "Snake", "Eagle", "Dolphin"],
                        "answer": 3,  # Dolphin (0-indexed)
                    },
                    metadata={"category": "biology", "difficulty": "medium"},
                ),
                DatasetEntry(
                    content={
                        "question": "What is the square root of 144?",
                        "choices": ["10", "12", "14", "16"],
                        "answer": 1,  # 12 (0-indexed)
                    },
                    metadata={"category": "mathematics", "difficulty": "easy"},
                ),
            ]


def basic_usage() -> None:
    """Demonstrate basic dataset usage with direct loading."""
    # Register and load our mock dataset
    register_mock_dataset()
    qa_dataset = datasets("mock_qa")
    print(f"Loaded mock QA dataset with {len(qa_dataset)} entries")

    # Print the first two entries for demonstration
    for i, entry in enumerate(qa_dataset):
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
        .sample(1)  # Take only one sample
        .seed(42)  # Set random seed for reproducibility
        .build("mock_qa")  # Build the configured dataset
    )

    print(f"\nLoaded 1 sample from mock QA dataset:")
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

        # Get category from metadata with fallback
        print(f"Category: {entry.metadata.get('category', 'unknown')}")
        print(f"Difficulty: {entry.metadata.get('difficulty', 'unknown')}")


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
    @register(
        "short_answer_qa",
        source="custom/short_answer",
        task_type=TaskType.SHORT_ANSWER,
        description="A custom dataset for short answer QA examples",
    )
    class ShortAnswerQADataset:
        """A custom short answer QA dataset for demonstration purposes."""

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
                DatasetEntry(
                    query="Who wrote Romeo and Juliet?",
                    metadata={
                        "category": "literature",
                        "answer": "William Shakespeare",
                    },
                ),
            ]

    # Now we can use our custom dataset
    try:
        short_answer_data = datasets("short_answer_qa")
        print(f"\nLoaded short answer dataset with {len(short_answer_data)} entries")

        for entry in short_answer_data:
            # Safely access fields with appropriate error handling
            print(f"\nQuestion: {entry.query}")
            print(f"Answer: {entry.metadata.get('answer', 'Unknown')}")
            print(f"Category: {entry.metadata.get('category', 'Uncategorized')}")
    except Exception as e:
        print(f"Error loading short answer dataset: {str(e)}")


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
