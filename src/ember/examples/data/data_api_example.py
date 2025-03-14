"""Example usage of the Ember data API.

This example demonstrates the intuitive API for working with datasets in Ember.

To run:
    poetry run python src/ember/examples/data_api_example.py
"""

import ember
from ember.api import datasets, DatasetBuilder, register, DatasetEntry, TaskType


def basic_usage():
    """Demonstrate basic dataset usage."""
    # Direct loading
    mmlu_dataset = datasets("mmlu")
    print(f"Loaded MMLU dataset with {len(mmlu_dataset)} entries")
    
    for i, entry in enumerate(mmlu_dataset):
        if i >= 2:  # Just show a couple of examples
            break
            
        print(f"\nEntry {i + 1}:")
        question = entry.content["question"]
        choices = entry.content["choices"]
        answer_idx = entry.content["answer"]
        
        print(f"Question: {question}")
        print("Choices:")
        for j, choice in enumerate(choices):
            print(f"  {j}. {choice}")
        print(f"Answer: {choices[answer_idx]} (index {answer_idx})")
        
        if entry.metadata:
            print("Metadata:")
            for key, value in entry.metadata.items():
                print(f"  {key}: {value}")


def advanced_usage():
    """Demonstrate advanced dataset usage with the builder pattern."""
    # Use the builder pattern for configuration
    dataset = (
        DatasetBuilder()
        .split("test")
        .sample(1)
        .seed(42)
        .build("mmlu")
    )
    
    print(f"\nLoaded 1 sample from MMLU test split:")
    if len(dataset) > 0:
        entry = dataset[0]
        print(f"Question: {entry.content['question']}")
        print("Choices:")
        for j, choice in enumerate(entry.content['choices']):
            print(f"  {j}. {choice}")
        print(f"Answer: {entry.content['choices'][entry.content['answer']]} (index {entry.content['answer']})")
        print(f"Domain: {entry.metadata.get('domain', 'unknown')}")


def cli_usage():
    """Demonstrate how to use the CLI."""
    print("\nCLI Usage:")
    print("To list all datasets:")
    print("  ember dataset list")
    print("\nTo show dataset info:")
    print("  ember dataset info mmlu")
    print("\nTo explore a dataset:")
    print("  ember dataset explore mmlu --count 5")
    print("\nTo export a dataset:")
    print("  ember dataset export mmlu --split test --count 10 --output mmlu.json")


def custom_dataset():
    """Demonstrate how to create a custom dataset."""
    # Register a custom dataset (for demonstration only)
    @register("custom_qa", source="custom/qa", task_type=TaskType.SHORT_ANSWER)
    class CustomQADataset:
        def __init__(self):
            pass
            
        def load(self, config=None):
            """Load the custom dataset."""
            # In a real implementation, this would load from a source
            return [
                DatasetEntry(
                    query="What is the capital of France?",
                    metadata={"category": "geography", "answer": "Paris"}
                ),
                DatasetEntry(
                    query="What is the largest mammal?",
                    metadata={"category": "biology", "answer": "Blue whale"}
                )
            ]
    
    # Now we can use our custom dataset
    custom_data = datasets("custom_qa")
    print(f"\nLoaded custom dataset with {len(custom_data)} entries")
    
    for entry in custom_data:
        print(f"\nQuestion: {entry.query}")
        print(f"Answer: {entry.metadata['answer']}")
        print(f"Category: {entry.metadata['category']}")


def main():
    """Run the data API example."""
    print("Ember Data API Example\n" + "=" * 24 + "\n")
    
    # Basic usage
    basic_usage()
    
    # Advanced usage
    advanced_usage()
    
    # CLI usage info
    cli_usage()
    
    # Custom dataset
    custom_dataset()
    
    print("\nFor more information, see the documentation in the project repository")


if __name__ == "__main__":
    main()