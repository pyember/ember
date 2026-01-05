"""Batch processing with vmap and parallel execution.

This example demonstrates:
- Using vmap for vectorized operations
- Parallel batch processing patterns
- Controlling batch axes and output shapes
- Memory-efficient batch iteration

Run with:
    python examples/06_performance_optimization/batch_processing.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from ember.api import op
from ember.xcs import Config, vmap


# =============================================================================
# Part 1: Basic vmap Usage
# =============================================================================

@op
def process_single(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single item."""
    value = item.get("value", 0)
    return {
        "original": value,
        "doubled": value * 2,
        "squared": value ** 2,
    }


def demonstrate_basic_vmap() -> None:
    """Show basic vmap usage for batch processing."""
    print("Part 1: Basic vmap Usage")
    print("-" * 50)

    # Single item processing
    single_result = process_single({"value": 5})
    print(f"Single item: {single_result}")
    print()

    # Batch processing with vmap
    process_batch = vmap(process_single)

    items = [
        {"value": 1},
        {"value": 2},
        {"value": 3},
        {"value": 4},
        {"value": 5},
    ]

    # vmap automatically maps across the batch
    results = process_batch(items)
    print("Batch results:")
    for i, result in enumerate(results):
        print(f"  Item {i}: {result}")
    print()


# =============================================================================
# Part 2: Manual Batch Processing Patterns
# =============================================================================

def process_batch_sequential(
    items: List[Dict[str, Any]],
    processor: Any,
) -> List[Dict[str, Any]]:
    """Process items sequentially."""
    return [processor(item) for item in items]


def process_batch_parallel(
    items: List[Dict[str, Any]],
    processor: Any,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Process items in parallel using threads."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(processor, item) for item in items]
        return [f.result() for f in futures]


def demonstrate_manual_batching() -> None:
    """Show manual batching patterns."""
    print("Part 2: Manual Batch Processing Patterns")
    print("-" * 50)

    items = [{"value": i} for i in range(10)]

    # Sequential processing
    start = time.perf_counter()
    sequential_results = process_batch_sequential(items, process_single)
    seq_time = time.perf_counter() - start
    print(f"Sequential: {len(sequential_results)} items in {seq_time * 1000:.2f}ms")

    # Parallel processing
    start = time.perf_counter()
    parallel_results = process_batch_parallel(items, process_single, max_workers=4)
    par_time = time.perf_counter() - start
    print(f"Parallel:   {len(parallel_results)} items in {par_time * 1000:.2f}ms")
    print()


# =============================================================================
# Part 3: vmap with Configuration
# =============================================================================

def demonstrate_vmap_config() -> None:
    """Show vmap configuration options."""
    print("Part 3: vmap with Configuration")
    print("-" * 50)

    @op
    def compute(x: int) -> int:
        return x * x

    # Configure parallelism
    config = Config(parallel=True, max_workers=4)
    parallel_compute = vmap(compute, config=config)

    inputs = list(range(10))
    results = parallel_compute(inputs)

    print("Parallel vmap results:")
    print(f"  Inputs:  {inputs}")
    print(f"  Outputs: {list(results)}")
    print()

    # Sequential vmap (for comparison)
    seq_config = Config(parallel=False)
    sequential_compute = vmap(compute, config=seq_config)

    seq_results = sequential_compute(inputs)
    print("Sequential vmap results:")
    print(f"  Outputs: {list(seq_results)}")
    print()


# =============================================================================
# Part 4: Batching Strategies
# =============================================================================

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def demonstrate_batching_strategies() -> None:
    """Show different batching strategies."""
    print("Part 4: Batching Strategies")
    print("-" * 50)

    # Large dataset simulation
    large_dataset = [{"value": i} for i in range(100)]

    # Strategy 1: Process all at once
    print("Strategy 1: All at once")
    process_all = vmap(process_single)
    all_results = process_all(large_dataset)
    print(f"  Processed {len(all_results)} items")

    # Strategy 2: Chunk into mini-batches
    print("\nStrategy 2: Mini-batches")
    chunk_size = 20
    chunks = chunk_list(large_dataset, chunk_size)
    print(f"  Split into {len(chunks)} chunks of size {chunk_size}")

    all_chunked_results = []
    for chunk in chunks:
        chunk_results = process_all(chunk)
        all_chunked_results.extend(chunk_results)
    print(f"  Total processed: {len(all_chunked_results)} items")

    # Strategy 3: Streaming with limits
    print("\nStrategy 3: Generator-based (memory efficient)")

    def process_streaming(items: List[Any], batch_size: int = 10):
        """Process items in a streaming fashion."""
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            yield from process_all(batch)

    streamed = list(process_streaming(large_dataset, batch_size=25))
    print(f"  Processed {len(streamed)} items via generator")
    print()


# =============================================================================
# Part 5: Handling Heterogeneous Batches
# =============================================================================

@op
def process_with_context(item: Dict[str, Any], context: str) -> Dict[str, Any]:
    """Process item with shared context."""
    return {
        "item": item.get("value", 0),
        "context": context,
        "combined": f"{context}_{item.get('value', 0)}",
    }


def demonstrate_heterogeneous_batches() -> None:
    """Show handling of batches with shared context."""
    print("Part 5: Batches with Shared Context")
    print("-" * 50)

    items = [{"value": i} for i in range(5)]
    shared_context = "batch_001"

    # Use in_axes to specify which args are batched
    # in_axes=0 means first arg is batched, None means broadcasted
    batch_with_context = vmap(process_with_context, in_axes=(0, None))

    results = batch_with_context(items, shared_context)
    print("Results with shared context:")
    for result in results:
        print(f"  {result}")
    print()


# =============================================================================
# Part 6: Batch Processing Best Practices
# =============================================================================

def demonstrate_best_practices() -> None:
    """Show batch processing best practices."""
    print("Part 6: Best Practices")
    print("-" * 50)

    practices = [
        (
            "Choose batch size carefully",
            "Too small: overhead dominates. Too large: memory issues.",
        ),
        (
            "Use vmap for homogeneous batches",
            "All items should have the same structure for vmap.",
        ),
        (
            "Consider memory constraints",
            "Chunk large datasets to avoid OOM errors.",
        ),
        (
            "Enable parallelism for I/O",
            "Use Config(parallel=True) for LLM calls.",
        ),
        (
            "Profile before optimizing",
            "Measure actual bottlenecks with get_jit_stats().",
        ),
    ]

    for title, description in practices:
        print(f"  {title}:")
        print(f"    {description}")
    print()


def main() -> None:
    """Demonstrate batch processing patterns."""
    print("Batch Processing")
    print("=" * 50)
    print()

    demonstrate_basic_vmap()
    demonstrate_manual_batching()
    demonstrate_vmap_config()
    demonstrate_batching_strategies()
    demonstrate_heterogeneous_batches()
    demonstrate_best_practices()

    print("Key Takeaways")
    print("-" * 50)
    print("1. vmap vectorizes functions across batch dimensions")
    print("2. Use Config(parallel=True) for concurrent execution")
    print("3. Chunk large datasets into mini-batches for memory efficiency")
    print("4. Use in_axes to control which arguments are batched")
    print("5. Profile to find optimal batch sizes for your workload")
    print()
    print("Next: See optimization_techniques.py for advanced patterns")


if __name__ == "__main__":
    main()
