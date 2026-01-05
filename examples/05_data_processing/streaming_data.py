"""Stream data lazily for memory-efficient processing.

This example demonstrates:
- Lazy evaluation with StreamIterator
- Composable pipeline operations
- Memory-efficient processing of large datasets
- Fluent API for data transformations

Run with:
    python examples/05_data_processing/streaming_data.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterator, List

from ember.api.data import (
    StreamIterator,
    register,
    stream,
)


# =============================================================================
# Part 1: Basic Streaming
# =============================================================================

class SimulatedLargeSource:
    """Simulate a large dataset that yields records on demand."""

    def __init__(self, total_records: int):
        self._total = total_records
        self.records_generated = 0

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches lazily, tracking generation."""
        batch: List[Dict[str, Any]] = []
        for i in range(self._total):
            self.records_generated += 1
            batch.append({
                "id": i,
                "question": f"Question number {i}",
                "answer": f"Answer {i}",
                "category": ["math", "science", "history"][i % 3],
                "difficulty": (i % 5) + 1,
            })
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def demonstrate_lazy_evaluation() -> None:
    """Show that streaming is truly lazy."""
    print("Part 1: Lazy Evaluation")
    print("-" * 50)

    # Create a source that tracks how many records it generates
    source = SimulatedLargeSource(1000)
    register("lazy_demo", source)

    # Create stream but don't iterate
    print("Creating stream (no iteration yet)...")
    iterator = stream("lazy_demo", normalize="none")
    print(f"Records generated so far: {source.records_generated}")

    # Now iterate over just 5 items
    print("\nIterating over 5 items...")
    count = 0
    for item in iterator:
        count += 1
        if count >= 5:
            break
    print(f"Records generated after 5 iterations: {source.records_generated}")
    print("(Only batch-size records generated, not all 1000)")
    print()


# =============================================================================
# Part 2: Fluent Pipeline API
# =============================================================================

def demonstrate_fluent_api() -> None:
    """Show the fluent API for building pipelines."""
    print("Part 2: Fluent Pipeline API")
    print("-" * 50)

    # Create test data
    test_data = [
        {"id": i, "value": i * 10, "category": ["A", "B", "C"][i % 3]}
        for i in range(20)
    ]

    class ListSource:
        def __init__(self, data: List[Dict[str, Any]]):
            self._data = data

        def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
            for i in range(0, len(self._data), batch_size):
                yield self._data[i : i + batch_size]

    register("fluent_demo", ListSource(test_data))

    # Build pipeline with fluent API
    print("Building pipeline: filter -> transform -> limit")
    pipeline = (
        stream("fluent_demo", normalize="none")
        .filter(lambda x: x["category"] == "A")  # Keep only category A
        .transform(lambda x: {**x, "doubled": x["value"] * 2})  # Add computed field
        .limit(3)  # Take first 3 results
    )

    print("Pipeline definition (no execution yet)")
    print("Type:", type(pipeline).__name__)
    print()

    # Execute pipeline
    print("Executing pipeline:")
    results = list(pipeline)
    for r in results:
        print(f"  id={r['id']}, value={r['value']}, doubled={r['doubled']}")
    print()


# =============================================================================
# Part 3: StreamIterator Methods
# =============================================================================

def demonstrate_iterator_methods() -> None:
    """Show various StreamIterator convenience methods."""
    print("Part 3: StreamIterator Methods")
    print("-" * 50)

    # Create sample data
    sample_data = [
        {"id": i, "score": i * 7 % 100}
        for i in range(50)
    ]

    class SampleSource:
        def __init__(self, data: List[Dict[str, Any]]):
            self._data = data

        def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
            for i in range(0, len(self._data), batch_size):
                yield self._data[i : i + batch_size]

    register("methods_demo", SampleSource(sample_data))

    # .first(n) - Get first N items as a list
    print(".first(3) - Get first 3 items:")
    first_three = stream("methods_demo", normalize="none").first(3)
    print(f"  Got {len(first_three)} items: {[x['id'] for x in first_three]}")
    print()

    # .limit(n) - Return a new iterator capped at N
    print(".limit(5) - Create limited iterator:")
    limited = stream("methods_demo", normalize="none").limit(5)
    count = sum(1 for _ in limited)
    print(f"  Iterator yielded {count} items")
    print()

    # .filter() - Chain multiple filters
    print("Chained .filter() calls:")
    filtered = (
        stream("methods_demo", normalize="none")
        .filter(lambda x: x["score"] > 50)  # High scores
        .filter(lambda x: x["id"] < 30)  # Low IDs
        .first(5)
    )
    print(f"  High scores (>50) with low IDs (<30): {[x['id'] for x in filtered]}")
    print()

    # .transform() - Chain transforms
    print("Chained .transform() calls:")
    transformed = (
        stream("methods_demo", normalize="none")
        .transform(lambda x: {**x, "normalized": x["score"] / 100})
        .transform(lambda x: {**x, "grade": "A" if x["normalized"] > 0.7 else "B"})
        .first(3)
    )
    for t in transformed:
        print(f"  id={t['id']}, score={t['score']}, grade={t['grade']}")
    print()


# =============================================================================
# Part 4: Normalization Modes
# =============================================================================

def demonstrate_normalization() -> None:
    """Show different normalization modes."""
    print("Part 4: Normalization Modes")
    print("-" * 50)

    # Create data with various field names
    raw_data = [
        {"prompt": "What is AI?", "response": "Artificial Intelligence"},
        {"query": "Define ML", "target": "Machine Learning"},
        {"question": "What is DL?", "answer": "Deep Learning"},
    ]

    class RawSource:
        def __init__(self, data: List[Dict[str, Any]]):
            self._data = data

        def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
            yield self._data

    register("normalize_demo", RawSource(raw_data))

    # normalize="none" - Raw data passthrough
    print('normalize="none" (raw passthrough):')
    for item in stream("normalize_demo", normalize="none"):
        print(f"  Keys: {list(item.keys())}")
    print()

    # normalize="dict" - Normalized dictionary
    print('normalize="dict" (normalized dict):')
    for item in stream("normalize_demo", normalize="dict"):
        print(f"  question={item['question'][:20]}, answer={item['answer'][:20]}")
    print()

    # normalize="record" (default) - DataRecord objects
    print('normalize="record" (DataRecord objects):')
    for record in stream("normalize_demo"):  # default is "record"
        print(f"  record.question.text = {record.question.text[:20]}")
    print()

    # Using .as_dicts() and .records() for conversion
    print("Converting between modes:")
    base = stream("normalize_demo", normalize="none")
    as_dicts = base.as_dicts().first(1)
    print(f"  .as_dicts() result keys: {list(as_dicts[0].keys())}")
    print()


# =============================================================================
# Part 5: Memory-Efficient Patterns
# =============================================================================

def demonstrate_memory_efficiency() -> None:
    """Show patterns for memory-efficient data processing."""
    print("Part 5: Memory-Efficient Patterns")
    print("-" * 50)

    # Simulate large dataset
    class LargeDataset:
        def __init__(self, size: int):
            self._size = size

        def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
            batch: List[Dict[str, Any]] = []
            for i in range(self._size):
                batch.append({"id": i, "data": f"record_{i}"})
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    register("large_dataset", LargeDataset(10000))

    # Pattern 1: Process in batches without loading all into memory
    print("Pattern 1: Stream processing (no full load)")
    total = 0
    for item in stream("large_dataset", max_items=100, normalize="none"):
        total += 1
    print(f"  Processed {total} items without loading all 10000")
    print()

    # Pattern 2: Early termination with limit
    print("Pattern 2: Early termination")
    results = stream("large_dataset", normalize="none").limit(10).first(10)
    print(f"  Got exactly {len(results)} items via .limit(10).first(10)")
    print()

    # Pattern 3: Filter before transform (reduce work)
    print("Pattern 3: Filter early (reduce downstream work)")
    efficient_pipeline = (
        stream("large_dataset", normalize="none")
        .filter(lambda x: x["id"] % 100 == 0)  # Filter first: 1% of data
        .transform(lambda x: {**x, "processed": True})  # Transform only filtered
        .limit(5)
    )
    processed = list(efficient_pipeline)
    print(f"  Filtered then transformed: {len(processed)} items")
    print(f"  IDs: {[p['id'] for p in processed]}")
    print()


# =============================================================================
# Part 6: Composable Pipelines
# =============================================================================

def create_preprocessing_pipeline(source_name: str) -> StreamIterator:
    """Factory function to create a reusable preprocessing pipeline."""
    return (
        stream(source_name, normalize="dict")
        .filter(lambda x: x.get("question"))  # Must have question
        .filter(lambda x: len(x.get("question", "")) > 5)  # Min length
        .transform(lambda x: {
            **x,
            "question": x["question"].strip().lower(),
            "processed": True,
        })
    )


def demonstrate_composable_pipelines() -> None:
    """Show how to build reusable, composable pipelines."""
    print("Part 6: Composable Pipelines")
    print("-" * 50)

    # Create test data
    test_data = [
        {"question": "What is Python?", "answer": "A language"},
        {"question": "Hi", "answer": "Too short"},
        {"question": "   How does ML work?   ", "answer": "It learns"},
        {"answer": "No question here"},
    ]

    class TestSource:
        def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
            yield test_data

    register("composable_demo", TestSource())

    # Use the factory function
    print("Using preprocessing pipeline factory:")
    pipeline = create_preprocessing_pipeline("composable_demo")
    results = pipeline.first(10)

    for r in results:
        print(f"  '{r['question']}' (processed={r['processed']})")
    print()

    # Extend the pipeline further
    print("Extending the pipeline:")
    extended = (
        create_preprocessing_pipeline("composable_demo")
        .transform(lambda x: {**x, "length": len(x["question"])})
        .limit(2)
    )
    for r in list(extended):
        print(f"  '{r['question']}' (length={r['length']})")
    print()


def main() -> None:
    """Demonstrate streaming data patterns."""
    print("Streaming Data Processing")
    print("=" * 50)
    print()

    demonstrate_lazy_evaluation()
    demonstrate_fluent_api()
    demonstrate_iterator_methods()
    demonstrate_normalization()
    demonstrate_memory_efficiency()
    demonstrate_composable_pipelines()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Streaming is lazy - data is only fetched when iterated")
    print("2. Use fluent API (.filter, .transform, .limit) to build pipelines")
    print("3. Filter early to reduce downstream processing")
    print("4. Use .first(n) for quick sampling, .limit(n) for capped iteration")
    print("5. Choose normalization mode based on downstream needs")
    print()
    print("Next: Explore examples/06_performance_optimization/ for JIT and batching")


if __name__ == "__main__":
    main()
