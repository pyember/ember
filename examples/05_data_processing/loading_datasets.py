"""Load and process datasets with the Ember data API.

This example demonstrates:
- Loading datasets from various sources
- Registering custom data sources
- Filtering and transforming records
- Working with DataRecord objects

Run with:
    python examples/05_data_processing/loading_datasets.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ember.api.data import (
    DataSource,
    FileSource,
    from_file,
    list_datasets,
    load,
    load_file,
    metadata,
    register,
    stream,
)


# =============================================================================
# Part 1: Listing Available Datasets
# =============================================================================

def demonstrate_discovery() -> None:
    """Show how to discover available datasets."""
    print("Part 1: Dataset Discovery")
    print("-" * 50)

    # List all registered datasets
    datasets = list_datasets()
    print(f"Registered datasets ({len(datasets)} total):")
    for name in datasets[:8]:  # Show first 8
        print(f"  - {name}")
    if len(datasets) > 8:
        print(f"  ... and {len(datasets) - 8} more")
    print()

    # Get metadata for a dataset
    if datasets:
        info = metadata(datasets[0])
        print(f"Metadata for '{info.name}':")
        print(f"  Description: {info.description}")
        print(f"  Streaming supported: {info.streaming_supported}")
    print()


# =============================================================================
# Part 2: Loading from Files
# =============================================================================

def demonstrate_file_loading() -> None:
    """Show how to load data from local files."""
    print("Part 2: Loading from Files")
    print("-" * 50)

    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create JSON file
        json_path = Path(tmpdir) / "questions.json"
        questions = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Capital of France?", "answer": "Paris"},
            {"question": "Largest planet?", "answer": "Jupiter"},
        ]
        json_path.write_text(json.dumps(questions))

        # Create JSONL file
        jsonl_path = Path(tmpdir) / "logs.jsonl"
        logs = [
            {"event": "login", "user": "alice"},
            {"event": "query", "user": "bob"},
            {"event": "logout", "user": "alice"},
        ]
        jsonl_path.write_text("\n".join(json.dumps(log) for log in logs))

        # Create CSV file
        csv_path = Path(tmpdir) / "scores.csv"
        csv_path.write_text("name,score,grade\nalice,95,A\nbob,87,B\ncharlie,92,A")

        # Load JSON file
        print("Loading JSON file:")
        records = load_file(json_path, normalize="dict")
        for r in records:
            print(f"  Q: {r['question']} -> A: {r['answer']}")
        print()

        # Stream JSONL file
        print("Streaming JSONL file:")
        for item in from_file(jsonl_path, normalize="none"):
            print(f"  Event: {item['event']} by {item['user']}")
        print()

        # Load and filter CSV
        print("Loading CSV with filter:")
        high_scores = load_file(
            csv_path,
            filter=lambda row: row.get("grade") == "A",
            normalize="none",
        )
        for row in high_scores:
            print(f"  {row['name']}: {row['score']} ({row['grade']})")
        print()


# =============================================================================
# Part 3: Custom Data Sources
# =============================================================================

class InMemorySource:
    """Custom data source backed by an in-memory list."""

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches from the in-memory store."""
        for i in range(0, len(self._records), batch_size):
            yield self._records[i : i + batch_size]


class GeneratorSource:
    """Custom data source that generates records on demand."""

    def __init__(self, count: int):
        self._count = count

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield generated records in batches."""
        batch: List[Dict[str, Any]] = []
        for i in range(self._count):
            batch.append({
                "id": i,
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "difficulty": (i % 3) + 1,
            })
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def demonstrate_custom_sources() -> None:
    """Show how to register and use custom data sources."""
    print("Part 3: Custom Data Sources")
    print("-" * 50)

    # Register an in-memory source
    sample_data = [
        {"question": "What is Python?", "answer": "A programming language"},
        {"question": "What is Ember?", "answer": "An AI framework"},
        {"question": "What is LLM?", "answer": "Large Language Model"},
    ]
    register("my_dataset", InMemorySource(sample_data))

    # Use it via stream()
    print("Custom in-memory source:")
    for record in stream("my_dataset", normalize="dict"):
        print(f"  {record['question']}")
    print()

    # Register a generator source
    register("generated", GeneratorSource(100))

    # Stream with limits
    print("Generated source (first 5):")
    for record in stream("generated", max_items=5, normalize="dict"):
        print(f"  ID {record['id']}: {record['question']}")
    print()


# =============================================================================
# Part 4: Filtering and Transforming
# =============================================================================

def demonstrate_filtering() -> None:
    """Show filtering and transformation capabilities."""
    print("Part 4: Filtering and Transforming")
    print("-" * 50)

    # Create test data
    test_data = [
        {"question": "Easy Q1", "difficulty": 1, "topic": "math"},
        {"question": "Hard Q2", "difficulty": 3, "topic": "science"},
        {"question": "Medium Q3", "difficulty": 2, "topic": "math"},
        {"question": "Easy Q4", "difficulty": 1, "topic": "history"},
        {"question": "Hard Q5", "difficulty": 3, "topic": "math"},
    ]
    register("quiz", InMemorySource(test_data))

    # Filter by difficulty
    print("Hard questions only (difficulty >= 3):")
    hard_questions = list(
        stream(
            "quiz",
            filter=lambda r: r.get("difficulty", 0) >= 3,
            normalize="none",
        )
    )
    for q in hard_questions:
        print(f"  {q['question']} (difficulty: {q['difficulty']})")
    print()

    # Filter by topic
    print("Math questions only:")
    math_questions = list(
        stream(
            "quiz",
            filter=lambda r: r.get("topic") == "math",
            normalize="none",
        )
    )
    for q in math_questions:
        print(f"  {q['question']}")
    print()

    # Transform records
    print("Transformed records (uppercase questions):")
    transformed = list(
        stream(
            "quiz",
            transform=lambda r: {**r, "question": r["question"].upper()},
            max_items=3,
            normalize="none",
        )
    )
    for q in transformed:
        print(f"  {q['question']}")
    print()


# =============================================================================
# Part 5: Working with DataRecord
# =============================================================================

def demonstrate_data_records() -> None:
    """Show how to work with DataRecord objects."""
    print("Part 5: Working with DataRecord")
    print("-" * 50)

    # Create test data with choices
    test_data = [
        {
            "question": "What is 2 + 2?",
            "answer": "B",
            "choices": ["3", "4", "5", "6"],
            "category": "arithmetic",
        },
        {
            "question": "Capital of Japan?",
            "answer": "A",
            "choices": ["Tokyo", "Osaka", "Kyoto"],
            "country": "Japan",
        },
    ]
    register("mcq", InMemorySource(test_data))

    # Stream as DataRecord (default normalization)
    print("DataRecord objects:")
    for record in stream("mcq"):  # normalize="record" is default
        print(f"  Question: {record.question.text}")
        print(f"  Answer: {record.answer.text}")
        print(f"  Choices: {len(record.choices)} options")
        if record.choices:
            for key, value in record.choices.items():
                print(f"    {key}: {value.text}")
        print(f"  Metadata: {record.metadata}")
        print()


# =============================================================================
# Part 6: Batch Loading
# =============================================================================

def demonstrate_batch_loading() -> None:
    """Show batch loading patterns."""
    print("Part 6: Batch Loading")
    print("-" * 50)

    # Register a larger dataset
    large_data = [
        {"id": i, "value": f"item_{i}"}
        for i in range(50)
    ]
    register("large", InMemorySource(large_data))

    # Load with limit using load()
    print("Eager loading (first 5 items):")
    items = load("large", max_items=5, normalize="none")
    print(f"  Loaded {len(items)} items: {[i['id'] for i in items]}")
    print()

    # Control batch size for memory efficiency
    print("Streaming with small batches:")
    count = 0
    for item in stream("large", batch_size=10, max_items=25, normalize="none"):
        count += 1
    print(f"  Processed {count} items with batch_size=10")
    print()


def main() -> None:
    """Demonstrate dataset loading patterns."""
    print("Loading Datasets")
    print("=" * 50)
    print()

    demonstrate_discovery()
    demonstrate_file_loading()
    demonstrate_custom_sources()
    demonstrate_filtering()
    demonstrate_data_records()
    demonstrate_batch_loading()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use list_datasets() and metadata() to discover available data")
    print("2. Load files with load_file() or stream with from_file()")
    print("3. Register custom sources by implementing read_batches()")
    print("4. Filter and transform data inline with stream()")
    print("5. DataRecord provides typed access to normalized fields")
    print()
    print("Next: See streaming_data.py for advanced streaming patterns")


if __name__ == "__main__":
    main()
