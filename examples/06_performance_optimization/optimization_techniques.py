"""Advanced optimization techniques with Ember XCS.

This example demonstrates:
- Combining jit, vmap, and other transforms
- Profiling and identifying bottlenecks
- Caching strategies for repeated operations
- Memory management patterns

Run with:
    python examples/06_performance_optimization/optimization_techniques.py
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from ember.api import op
from ember.xcs import Config, get_jit_stats, jit, scan, vmap


# =============================================================================
# Part 1: Combining Transforms
# =============================================================================

@op
def transform_item(x: int) -> int:
    """Simple transformation."""
    return x * 2 + 1


def demonstrate_combined_transforms() -> None:
    """Show combining multiple XCS transforms."""
    print("Part 1: Combining Transforms")
    print("-" * 50)

    # Basic usage
    print("Single transform:")
    print(f"  transform_item(5) = {transform_item(5)}")

    # JIT + vmap: Compile and vectorize
    @jit
    def batch_transform(items: List[int]) -> List[int]:
        """JIT-compiled batch transformation."""
        return [transform_item(x) for x in items]

    items = [1, 2, 3, 4, 5]
    print(f"\nJIT batch transform:")
    print(f"  Input:  {items}")
    print(f"  Output: {batch_transform(items)}")

    # vmap for automatic batching
    vmapped = vmap(transform_item)
    print(f"\nvmap transform:")
    print(f"  Output: {list(vmapped(items))}")
    print()


# =============================================================================
# Part 2: Scan for Sequential Operations
# =============================================================================

def demonstrate_scan() -> None:
    """Show scan for sequential operations with carry."""
    print("Part 2: Scan for Sequential Operations")
    print("-" * 50)

    # Running sum using scan
    @scan
    def running_sum(carry: int, x: int) -> Tuple[int, int]:
        """Compute running sum: (carry, current) -> (new_carry, output)."""
        new_carry = carry + x
        return new_carry, new_carry

    items = [1, 2, 3, 4, 5]
    final_carry, outputs = running_sum(0, items)

    print("Running sum with scan:")
    print(f"  Items: {items}")
    print(f"  Running sums: {list(outputs)}")
    print(f"  Final total: {final_carry}")
    print()

    # More complex: accumulating state
    @scan
    def accumulate_state(
        state: Dict[str, int], item: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Accumulate state across items."""
        new_count = state["count"] + 1
        new_sum = state["sum"] + item.get("value", 0)
        new_state = {"count": new_count, "sum": new_sum}
        output = {
            "step": new_count,
            "running_avg": new_sum / new_count,
        }
        return new_state, output

    records = [{"value": 10}, {"value": 20}, {"value": 15}, {"value": 25}]
    initial_state = {"count": 0, "sum": 0}
    final_state, step_outputs = accumulate_state(initial_state, records)

    print("Accumulating state with scan:")
    print(f"  Final state: {final_state}")
    print(f"  Step outputs: {list(step_outputs)}")
    print()


# =============================================================================
# Part 3: Caching Strategies
# =============================================================================

# Python-level caching
@lru_cache(maxsize=128)
def cached_expensive_computation(n: int) -> int:
    """Cache expensive computation results."""
    # Simulate expensive operation
    result = sum(i * i for i in range(n))
    return result


# JIT caching (automatic)
@jit
def jit_cached_computation(n: int) -> int:
    """JIT automatically caches execution graphs."""
    return sum(i * i for i in range(n))


def demonstrate_caching() -> None:
    """Show different caching strategies."""
    print("Part 3: Caching Strategies")
    print("-" * 50)

    # Python lru_cache
    print("Python lru_cache:")
    start = time.perf_counter()
    _ = cached_expensive_computation(1000)
    first_call = time.perf_counter() - start

    start = time.perf_counter()
    _ = cached_expensive_computation(1000)
    cached_call = time.perf_counter() - start

    print(f"  First call:  {first_call * 1000:.3f}ms")
    print(f"  Cached call: {cached_call * 1000:.3f}ms")
    print(f"  Speedup: {first_call / max(cached_call, 0.0001):.1f}x")
    print()

    # JIT caching
    print("JIT caching:")
    # Warmup
    _ = jit_cached_computation(100)
    stats1 = jit_cached_computation.stats()  # type: ignore[attr-defined]

    # Second call - uses cached graph
    _ = jit_cached_computation(100)
    stats2 = jit_cached_computation.stats()  # type: ignore[attr-defined]

    print(f"  After 1st call - misses: {stats1.get('misses', 0)}, hits: {stats1.get('hits', 0)}")
    print(f"  After 2nd call - misses: {stats2.get('misses', 0)}, hits: {stats2.get('hits', 0)}")
    print()


# =============================================================================
# Part 4: Profiling and Bottleneck Identification
# =============================================================================

def demonstrate_profiling() -> None:
    """Show profiling techniques."""
    print("Part 4: Profiling")
    print("-" * 50)

    @jit(config=Config(profile=True))
    def profiled_function(items: List[int]) -> int:
        """A function with profiling enabled."""
        return sum(x * x for x in items)

    # Run several times
    for _ in range(5):
        profiled_function(list(range(100)))

    # Get profiling stats
    stats = profiled_function.stats()  # type: ignore[attr-defined]
    print("Profiling results:")
    print(f"  Status: {stats.get('status', 'unknown')}")
    print(f"  Cache hits: {stats.get('hits', 0)}")
    print(f"  Cache misses: {stats.get('misses', 0)}")
    if "estimated_speedup" in stats:
        print(f"  Estimated speedup: {stats['estimated_speedup']:.2f}x")
    print()

    # Global profiler stats
    global_stats = get_jit_stats()
    print("Global profiler summary:")
    for key, value in list(global_stats.items())[:5]:
        print(f"  {key}: {value}")
    print()


# =============================================================================
# Part 5: Memory Management
# =============================================================================

def demonstrate_memory_management() -> None:
    """Show memory-efficient patterns."""
    print("Part 5: Memory Management")
    print("-" * 50)

    # Generator-based processing (lazy evaluation)
    def process_large_dataset_lazy(size: int):
        """Process items one at a time using a generator."""
        for i in range(size):
            yield {"id": i, "processed": i * 2}

    print("Lazy processing (generator):")
    lazy_results = process_large_dataset_lazy(1000)
    first_five = [next(lazy_results) for _ in range(5)]
    print(f"  First 5 items: {first_five}")
    print("  (Remaining items not computed until needed)")
    print()

    # Chunked processing
    def process_in_chunks(size: int, chunk_size: int = 100):
        """Process in chunks to limit memory usage."""
        for start in range(0, size, chunk_size):
            end = min(start + chunk_size, size)
            chunk = list(range(start, end))
            # Process chunk
            yield [x * 2 for x in chunk]

    print("Chunked processing:")
    chunks = list(process_in_chunks(500, chunk_size=100))
    print(f"  Processed in {len(chunks)} chunks")
    print(f"  First chunk size: {len(chunks[0])}")
    print()


# =============================================================================
# Part 6: Optimization Guidelines
# =============================================================================

def demonstrate_guidelines() -> None:
    """Show optimization guidelines."""
    print("Part 6: Optimization Guidelines")
    print("-" * 50)

    guidelines = [
        (
            "Measure first",
            "Profile before optimizing. Use get_jit_stats() to identify bottlenecks.",
        ),
        (
            "Use @jit for repeated operations",
            "JIT compilation amortizes over many calls. Most beneficial for LLM orchestration.",
        ),
        (
            "Use vmap for batch independence",
            "When items can be processed independently, vmap enables parallelism.",
        ),
        (
            "Use scan for sequential dependencies",
            "When you need to carry state across iterations.",
        ),
        (
            "Layer caching appropriately",
            "lru_cache for Python objects, JIT for execution graphs.",
        ),
        (
            "Manage memory with generators",
            "Use lazy evaluation for large datasets.",
        ),
        (
            "Tune parallelism",
            "Config(max_workers=N) to control thread pool size.",
        ),
    ]

    for i, (title, description) in enumerate(guidelines, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
    print()


def main() -> None:
    """Demonstrate optimization techniques."""
    print("Optimization Techniques")
    print("=" * 50)
    print()

    demonstrate_combined_transforms()
    demonstrate_scan()
    demonstrate_caching()
    demonstrate_profiling()
    demonstrate_memory_management()
    demonstrate_guidelines()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Combine jit + vmap for compiled batch operations")
    print("2. Use scan for sequential operations with state")
    print("3. Layer caching: lru_cache for results, JIT for graphs")
    print("4. Profile with Config(profile=True) and get_jit_stats()")
    print("5. Use generators and chunking for memory efficiency")
    print()
    print("Next: Explore examples/07_error_handling/ for robust patterns")


if __name__ == "__main__":
    main()
