"""Use JIT compilation with zero configuration.

This example demonstrates:
- The @jit decorator for automatic optimization
- Tracing semantics and what gets compiled
- Performance benefits of JIT
- When to use (and not use) JIT

Run with:
    python examples/03_simplified_apis/zero_config_jit.py
"""

from __future__ import annotations

import time

from ember.api import op
from ember.xcs import jit


# =============================================================================
# Part 1: Basic JIT Usage
# =============================================================================

@op
def slow_computation(n: int) -> int:
    """A computation that benefits from optimization."""
    result = 0
    for i in range(n):
        result += i * i
    return result


# JIT-compiled version - just add @jit
@jit
@op
def fast_computation(n: int) -> int:
    """Same computation, JIT-compiled."""
    result = 0
    for i in range(n):
        result += i * i
    return result


# =============================================================================
# Part 2: JIT with Complex Operations
# =============================================================================

@jit
@op
def process_text_pipeline(texts: list[str]) -> dict:
    """A multi-step text processing pipeline.

    JIT traces through all operations and optimizes the whole pipeline.
    """
    results = []
    for text in texts:
        # Step 1: Clean
        cleaned = text.strip().lower()
        # Step 2: Tokenize
        tokens = cleaned.split()
        # Step 3: Count
        count = len(tokens)
        results.append({"text": text, "tokens": count})

    return {
        "processed": len(results),
        "total_tokens": sum(r["tokens"] for r in results),
        "results": results[:3],  # Sample
    }


# =============================================================================
# Part 3: Understanding Tracing
# =============================================================================

@jit
@op
def traced_function(x: int, debug: bool = False) -> dict:
    """Demonstrates tracing behavior.

    Note: The 'debug' flag's VALUE at first call is traced.
    Subsequent calls with different values may not re-trace.
    """
    result = x * 2

    # This branch is traced based on first call's debug value
    if debug:
        return {"result": result, "debug_info": f"Input was {x}"}

    return {"result": result}


# =============================================================================
# Part 4: Performance Comparison
# =============================================================================

def benchmark(func, *args, iterations: int = 100) -> float:
    """Run a function multiple times and return average time."""
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def demonstrate_performance() -> None:
    """Compare JIT vs non-JIT performance."""
    print("Performance Comparison")
    print("-" * 50)

    n = 1000
    iterations = 100

    # Warm up JIT (first call compiles)
    fast_computation(n)

    # Benchmark
    slow_time = benchmark(slow_computation, n, iterations=iterations)
    fast_time = benchmark(fast_computation, n, iterations=iterations)

    print(f"Iterations: {iterations}, n={n}")
    print(f"Without JIT: {slow_time*1000:.3f}ms per call")
    print(f"With JIT:    {fast_time*1000:.3f}ms per call")

    if slow_time > 0:
        speedup = slow_time / max(fast_time, 1e-9)
        print(f"Speedup:     {speedup:.1f}x")
    print()


def main() -> None:
    """Demonstrate zero-config JIT."""
    print("Zero-Config JIT Compilation")
    print("=" * 50)
    print()

    # Part 1: Basic usage
    print("Part 1: Basic JIT Usage")
    print("-" * 50)
    result = fast_computation(100)
    print(f"JIT result: {result}")
    print()

    # Part 2: Complex pipelines
    print("Part 2: JIT with Pipelines")
    print("-" * 50)
    texts = ["Hello World", "Ember JIT", "Fast execution"]
    result = process_text_pipeline(texts)
    print(f"Pipeline result: {result}")
    print()

    # Part 3: Tracing behavior
    print("Part 3: Tracing Behavior")
    print("-" * 50)
    result1 = traced_function(5, debug=False)
    result2 = traced_function(10, debug=False)
    print(f"First call:  {result1}")
    print(f"Second call: {result2}")
    print("Note: JIT traces control flow on first call")
    print()

    # Part 4: Performance
    demonstrate_performance()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Add @jit decorator for automatic optimization")
    print("2. JIT traces on first call, caches for subsequent calls")
    print("3. Best for: repeated calls, pure computations")
    print("4. Avoid for: dynamic control flow, side effects")
    print()
    print("Next: See natural_api_showcase.py for ergonomic patterns")


if __name__ == "__main__":
    main()
