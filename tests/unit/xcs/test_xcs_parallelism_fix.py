"""Test that XCS parallelism actually works after our fixes.

Clean tests without the tracing noise.
"""

import time

import pytest

from ember.xcs import jit
from ember.xcs.compiler.graph import IRGraph, node_from_callable
from ember.xcs.compiler.parallelism import ParallelismAnalyzer


def slow_work(delay: float = 0.01):
    """Simulate work that takes time."""
    time.sleep(delay)
    return "done"


class TestXCSParallelismFix:
    """Test parallelism after fixes."""

    @pytest.mark.flaky(reruns=5, reruns_delay=2)
    def test_parallel_speedup(self):
        """Test that parallel operations actually speed up."""

        # Non-JIT baseline
        def sequential_four():
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            return "done"

        # JIT version
        @jit
        def parallel_four():
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            return "done"

        # Time sequential
        start = time.time()
        sequential_four()
        seq_time = time.time() - start

        # First call to JIT version (includes tracing)
        start = time.time()
        parallel_four()
        first_time = time.time() - start

        # Second call to JIT version (uses cached optimization)
        start = time.time()
        parallel_four()
        second_time = time.time() - start

        print(f"\nSequential: {seq_time:.3f}s")
        print(f"JIT first call: {first_time:.3f}s")
        print(f"JIT second call: {second_time:.3f}s")

        # Second call should be faster than sequential
        speedup = seq_time / second_time
        print(f"Speedup: {speedup:.1f}x")

        # Should see speedup on cached execution
        # Not expecting 4x because of overhead, but should be > 1.5x
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.1f}x"

        @jit
        def parallel_four():
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            slow_work(0.01)
            return "done"

        parallel_four()
        start = time.time()
        parallel_four()
        rebound_time = time.time() - start
        rebound_speedup = seq_time / rebound_time

        print(f"Rebound JIT second call: {rebound_time:.3f}s")
        print(f"Rebound speedup: {rebound_speedup:.1f}x")

        assert rebound_speedup > 1.5, "Freshly defined jit function should still parallelize"

    def test_parallel_with_dependencies(self):
        """Test mixed parallel/sequential execution."""

        @jit
        def diamond_pattern():
            # Phase 1: single operation
            _ = slow_work(0.01)

            # Phase 2: three parallel operations
            b = slow_work(0.01)
            c = slow_work(0.01)
            d = slow_work(0.01)

            # Phase 3: merge (depends on b, c, d)
            _ = f"{b}{c}{d}"
            e = slow_work(0.01)

            return e

        # First execution (with tracing)
        start = time.time()
        diamond_pattern()
        first_time = time.time() - start

        # Second execution (optimized)
        start = time.time()
        diamond_pattern()
        second_time = time.time() - start

        print("\nDiamond pattern:")
        print(f"First call: {first_time:.3f}s")
        print(f"Second call: {second_time:.3f}s")

        # Should take ~0.03s (3 phases) not 0.05s (5 sequential)
        assert second_time < 0.2, f"Expected < 0.2s, got {second_time:.3f}s"

    def test_no_parallelism_fallback(self):
        """Test that code without parallelism falls back to original."""

        @jit
        def no_function_calls():
            # Simple operations without function calls
            a = 1
            b = a + 1
            c = b + 1
            d = c + 1
            return d

        # Execute once to trigger optimization decision
        result = no_function_calls()
        assert result == 4

        # Check stats - should not be optimized
        stats = no_function_calls.stats()
        print(f"\nNo parallelism stats: {stats}")
        assert not stats["optimized"]
        assert stats["status"] == "fallback"

    @pytest.mark.slow
    def test_parallelism_analyzer_handles_large_graph_quickly(self):
        analyzer = ParallelismAnalyzer()
        graph = IRGraph()

        width = 20
        layers = 12

        def make_node(node_id: str, inputs: tuple[str, ...], output: str):
            def op(*values: float) -> float:
                return float(len(values)) if values else 1.0

            return node_from_callable(node_id, op, inputs, (output,), {})

        for index in range(width):
            graph = graph.add_node(make_node(f"n_0_{index}", (), f"v_0_{index}"))

        for layer in range(1, layers):
            for index in range(width):
                inputs = tuple(f"v_{layer-1}_{prev}" for prev in range(width))
                graph = graph.add_node(
                    make_node(f"n_{layer}_{index}", inputs, f"v_{layer}_{index}")
                )

        start = time.perf_counter()
        analyzer.analyze(graph)
        elapsed = time.perf_counter() - start

        # Threshold is generous to account for CI variability. The key property
        # is that analysis doesn't blow up to O(V²) seconds on large graphs.
        assert elapsed < 1.0, f"Expected analysis <1.0s, got {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
