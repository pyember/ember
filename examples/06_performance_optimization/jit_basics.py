"""Just-In-Time compilation with Ember XCS.

This example demonstrates:
- The @jit decorator for automatic optimization
- How JIT traces and caches execution graphs
- Inspecting compilation statistics
- Configuration options for JIT behavior

Run with:
    python examples/06_performance_optimization/jit_basics.py
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from ember.api import op
from ember.xcs import Config, get_jit_stats, jit


# =============================================================================
# Part 1: Basic JIT Usage
# =============================================================================

@op
def simple_computation(x: int, y: int) -> int:
    """A simple computation."""
    return x * y + x


@jit
def jit_computation(x: int, y: int) -> int:
    """Same computation with JIT optimization."""
    return x * y + x


def demonstrate_basic_jit() -> None:
    """Show basic JIT decorator usage."""
    print("Part 1: Basic JIT Usage")
    print("-" * 50)

    # Both produce the same result
    result_plain = simple_computation(10, 5)
    result_jit = jit_computation(10, 5)

    print(f"Plain:  10 * 5 + 10 = {result_plain}")
    print(f"JIT:    10 * 5 + 10 = {result_jit}")
    print(f"Results match: {result_plain == result_jit}")
    print()


# =============================================================================
# Part 2: JIT with Complex Functions
# =============================================================================

@jit
def process_data(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a list of items with JIT optimization.

    JIT traces the execution graph and can parallelize
    independent operations.
    """
    total = 0
    categories: Dict[str, int] = {}

    for item in items:
        value = item.get("value", 0)
        category = item.get("category", "unknown")

        total += value
        categories[category] = categories.get(category, 0) + 1

    return {
        "total": total,
        "count": len(items),
        "average": total / len(items) if items else 0,
        "categories": categories,
    }


def demonstrate_complex_jit() -> None:
    """Show JIT with more complex operations."""
    print("Part 2: JIT with Complex Functions")
    print("-" * 50)

    # Create sample data
    items = [
        {"value": 10, "category": "A"},
        {"value": 20, "category": "B"},
        {"value": 15, "category": "A"},
        {"value": 25, "category": "C"},
        {"value": 30, "category": "B"},
    ]

    result = process_data(items)
    print(f"Total: {result['total']}")
    print(f"Count: {result['count']}")
    print(f"Average: {result['average']}")
    print(f"Categories: {result['categories']}")
    print()


# =============================================================================
# Part 3: JIT Statistics
# =============================================================================

@jit
def tracked_function(x: int) -> int:
    """Function with stats tracking."""
    return x ** 2 + x + 1


def demonstrate_jit_stats() -> None:
    """Show how to inspect JIT compilation statistics."""
    print("Part 3: JIT Statistics")
    print("-" * 50)

    # Cold start - first call traces the function
    print("Cold start (first call):")
    result1 = tracked_function(5)
    stats1 = tracked_function.stats()  # type: ignore[attr-defined]
    print(f"  Result: {result1}")
    print(f"  Status: {stats1.get('status', 'unknown')}")
    print(f"  Cache hits: {stats1.get('hits', 0)}")
    print(f"  Cache misses: {stats1.get('misses', 0)}")
    print()

    # Warm call - uses cached graph
    print("Warm call (second call with same signature):")
    result2 = tracked_function(10)
    stats2 = tracked_function.stats()  # type: ignore[attr-defined]
    print(f"  Result: {result2}")
    print(f"  Status: {stats2.get('status', 'unknown')}")
    print(f"  Cache hits: {stats2.get('hits', 0)}")
    print(f"  Cache misses: {stats2.get('misses', 0)}")
    print()

    # Global stats
    print("Global profiler stats:")
    global_stats = get_jit_stats()
    print(f"  Functions tracked: {len(global_stats.get('functions', {}))}")
    print()


# =============================================================================
# Part 4: JIT Configuration
# =============================================================================

def demonstrate_jit_config() -> None:
    """Show JIT configuration options."""
    print("Part 4: JIT Configuration")
    print("-" * 50)

    # Create custom configuration
    config = Config(
        cache=True,        # Enable graph caching
        parallel=True,     # Enable parallel execution
        max_workers=4,     # Maximum worker threads
        profile=True,      # Enable profiling
    )

    @jit(config=config)
    def configured_function(x: int) -> int:
        return x * 2

    result = configured_function(21)
    print(f"With custom config: {result}")
    print()

    # Show config options
    print("Available Config options:")
    print("  cache: bool      - Enable/disable graph caching")
    print("  parallel: bool   - Enable/disable parallel execution")
    print("  max_workers: int - Maximum concurrent workers")
    print("  profile: bool    - Enable performance profiling")
    print()


# =============================================================================
# Part 5: JIT Performance Comparison
# =============================================================================

def demonstrate_performance() -> None:
    """Compare performance with and without JIT."""
    print("Part 5: Performance Comparison")
    print("-" * 50)

    def heavy_computation(n: int) -> int:
        """Simulate a heavier computation."""
        result = 0
        for i in range(n):
            result += i * i
        return result

    @jit
    def jit_heavy_computation(n: int) -> int:
        """Same computation with JIT."""
        result = 0
        for i in range(n):
            result += i * i
        return result

    n = 1000

    # Time without JIT
    start = time.perf_counter()
    for _ in range(100):
        heavy_computation(n)
    plain_time = time.perf_counter() - start

    # Time with JIT (including warmup)
    jit_heavy_computation(n)  # Warmup
    start = time.perf_counter()
    for _ in range(100):
        jit_heavy_computation(n)
    jit_time = time.perf_counter() - start

    print(f"Plain execution (100 runs): {plain_time * 1000:.2f}ms")
    print(f"JIT execution (100 runs):   {jit_time * 1000:.2f}ms")
    print()

    # Note: JIT benefits are more pronounced with LLM orchestration
    print("Note: JIT provides significant benefits when:")
    print("  - Orchestrating multiple LLM calls")
    print("  - Running parallel operations")
    print("  - Reusing computation graphs")
    print()


# =============================================================================
# Part 6: When to Use JIT
# =============================================================================

def demonstrate_when_to_use() -> None:
    """Show when JIT is beneficial."""
    print("Part 6: When to Use JIT")
    print("-" * 50)

    use_cases = [
        ("LLM Orchestration", "Parallelizes independent LLM calls automatically"),
        ("Data Pipelines", "Caches and optimizes repeated transformations"),
        ("Ensemble Patterns", "Runs multiple models concurrently"),
        ("Batch Processing", "Efficiently processes items in parallel"),
    ]

    print("Good use cases for @jit:")
    for name, description in use_cases:
        print(f"  {name}:")
        print(f"    {description}")
    print()

    not_beneficial = [
        ("Single LLM Call", "No parallelism to exploit"),
        ("I/O Bound Operations", "Already limited by external resources"),
        ("Small Computations", "Tracing overhead may exceed savings"),
    ]

    print("Less beneficial for:")
    for name, reason in not_beneficial:
        print(f"  {name}: {reason}")
    print()


def main() -> None:
    """Demonstrate JIT basics."""
    print("JIT Compilation Basics")
    print("=" * 50)
    print()

    demonstrate_basic_jit()
    demonstrate_complex_jit()
    demonstrate_jit_stats()
    demonstrate_jit_config()
    demonstrate_performance()
    demonstrate_when_to_use()

    print("Key Takeaways")
    print("-" * 50)
    print("1. @jit traces functions and caches execution graphs")
    print("2. Subsequent calls reuse cached graphs for efficiency")
    print("3. Use .stats() to inspect compilation status")
    print("4. Configure with Config() for custom behavior")
    print("5. Most beneficial for LLM orchestration and parallelism")
    print()
    print("Next: See batch_processing.py for vmap and batching")


if __name__ == "__main__":
    main()
