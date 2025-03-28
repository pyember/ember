"""Example of a standalone dataset in Ember.

This example demonstrates how to create and use datasets without
relying on the full Ember data API infrastructure.

To run:
    uv run python src/ember/examples/data/data_api_example.py
"""

from typing import Dict, List


class DatasetEntry:
    """A simple dataset entry with questions and choices."""

    def __init__(
        self, query: str, choices: Dict[str, str], metadata: Dict[str, str]
    ) -> None:
        """Initialize a dataset entry.

        Args:
            query: The question text
            choices: A dictionary of lettered choices
            metadata: Additional information
        """
        self.query = query
        self.choices = choices
        self.metadata = metadata


def create_mock_dataset() -> List[DatasetEntry]:
    """Create a mock multiple-choice dataset."""
    return [
        DatasetEntry(
            query="What is the capital of France?",
            choices={"A": "Berlin", "B": "Madrid", "C": "Paris", "D": "Rome"},
            metadata={"correct_answer": "C", "category": "Geography"},
        ),
        DatasetEntry(
            query="Which of these is a mammal?",
            choices={"A": "Shark", "B": "Snake", "C": "Eagle", "D": "Dolphin"},
            metadata={"correct_answer": "D", "category": "Biology"},
        ),
        DatasetEntry(
            query="What is the square root of 144?",
            choices={"A": "10", "B": "12", "C": "14", "D": "16"},
            metadata={"correct_answer": "B", "category": "Mathematics"},
        ),
    ]


def show_dataset(entries: List[DatasetEntry]) -> None:
    """Display the contents of a dataset.

    Args:
        entries: List of dataset entries to display
    """
    print(f"\nDataset with {len(entries)} entries:")

    for i, entry in enumerate(entries):
        print(f"\nQuestion {i+1}: {entry.query}")
        print(f"Category: {entry.metadata.get('category', 'Unknown')}")

        print("Options:")
        correct = entry.metadata.get("correct_answer", "")
        for letter, text in entry.choices.items():
            is_correct = "✓" if letter == correct else " "
            print(f"  {letter}. {text} {is_correct}")


def sample_dataset(entries: List[DatasetEntry], count: int = 1) -> List[DatasetEntry]:
    """Sample a random subset from the dataset.

    Args:
        entries: The dataset to sample from
        count: Number of entries to sample

    Returns:
        A subset of the original dataset
    """
    import random

    # Return either the requested count or the max available
    count = min(count, len(entries))

    # Make a copy to avoid modifying the original
    sample = random.sample(entries, count)
    return sample


def main() -> None:
    """Run the dataset example."""
    print("Ember Dataset Example")
    print("====================")

    # Create the mock dataset
    dataset = create_mock_dataset()

    # Display all entries
    show_dataset(dataset)

    # Demonstrate sampling
    print("\nSampling Example:")
    sample = sample_dataset(dataset, count=1)
    print(f"Sampled 1 random entry: {sample[0].query}")

    # Show how you might process the data
    print("\nData Processing Example:")
    for entry in dataset:
        # Get the category and correct answer
        category = entry.metadata.get("category", "Unknown")
        correct_answer = entry.metadata.get("correct_answer", "")

        # Get the text of the correct answer
        if correct_answer in entry.choices:
            answer_text = entry.choices[correct_answer]
            print(f"• {entry.query} ({category}) → {answer_text}")


if __name__ == "__main__":
    main()
