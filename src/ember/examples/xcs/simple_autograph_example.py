"""Simple Auto Graph Example with Mock Operators.

This example demonstrates the enhanced JIT API with automatic graph building
using simple mock operators that don't require external dependencies.
It uses the new ember.api package structure.

To run:
    uv run python src/ember/examples/xcs/simple_autograph_example.py
"""

import logging
import time
from typing import Any, Dict

from ember.xcs import jit, execution_options, JITMode, get_jit_stats
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification

###############################################################################
# Mock Operators
###############################################################################


@jit(mode=JITMode.ENHANCED)
class AddOperator(Operator):
    """Simple operator that adds a value to the input."""

    specification = Specification(input_model=None, structured_output=None)

    def __init__(self, *, value: int = 1) -> None:
        self.value = value

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs.get("value", 0) + self.value
        return {"value": result}


@jit()
class MultiplyOperator(Operator):
    """Simple operator that multiplies the input by a value."""

    specification = Specification(input_model=None, structured_output=None)

    def __init__(self, *, value: int = 2) -> None:
        self.value = value

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs.get("value", 0) * self.value
        return {"value": result}


@jit()
class DelayOperator(Operator):
    """Simple operator that introduces a delay."""

    specification = Specification(input_model=None, structured_output=None)

    def __init__(self, *, delay: float = 0.1) -> None:
        self.delay = delay

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(self.delay)
        return inputs


###############################################################################
# Pipeline with Auto Graph Building
###############################################################################


@jit(sample_input={"value": 1})
class CalculationPipeline(Operator):
    """Pipeline that demonstrates automatic graph building.

    This pipeline composes multiple operators but doesn't require
    manual graph construction. The @jit decorator handles this
    automatically, building a graph based on the actual execution trace.
    """

    specification = Specification(input_model=None, structured_output=None)

    def __init__(
        self,
        *,
        add_value: int = 5,
        multiply_value: int = 2,
        num_delay_ops: int = 3,
        delay: float = 0.1,
    ) -> None:
        """Initialize the pipeline with configurable parameters."""
        self.add_op = AddOperator(value=add_value)
        self.multiply_op = MultiplyOperator(value=multiply_value)

        # Create multiple delay operators to demonstrate parallel execution
        self.delay_ops = [DelayOperator(delay=delay) for _ in range(num_delay_ops)]

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline on the given inputs."""
        # First, add
        added = self.add_op(inputs=inputs)

        # Then, apply delays in "parallel" (in a real scenario, these would be executed concurrently)
        delay_results = []
        for op in self.delay_ops:
            delay_results.append(op(inputs=added))

        # Finally, multiply
        return self.multiply_op(inputs=added)


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of automatic graph building."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Automatic Graph Building Example ===\n")

    # Create the pipeline
    pipeline = CalculationPipeline(
        add_value=10, multiply_value=3, num_delay_ops=5, delay=0.1
    )

    # Example inputs to demonstrate caching and reuse
    inputs = [{"value": 5}, {"value": 10}, {"value": 15}]

    print("First run - expect graph building overhead:")
    first_run_times = []

    for i, inp in enumerate(inputs):
        print(f"\nInput {i+1}: {inp}")

        start_time = time.perf_counter()
        result = pipeline(inputs=inp)
        elapsed = time.perf_counter() - start_time
        first_run_times.append(elapsed)

        # Show calculation steps
        input_value = inp["value"]
        expected_value = (input_value + 10) * 3  # add_value=10, multiply_value=3

        print(f"Result: {result}")
        print(f"Value: {result['value']}")
        print(
            f"Expected calculation: {input_value} + 10 = {input_value + 10}, then × 3 = {expected_value}"
        )
        print(f"Time: {elapsed:.4f}s")

    # Cached execution demonstration
    print("\nRepeat first input to demonstrate cached execution:")
    start_time = time.perf_counter()
    result = pipeline(inputs=inputs[0])
    cached_time = time.perf_counter() - start_time

    print(f"Result: {result}")
    print(f"Value: {result['value']}")
    print(f"Time: {cached_time:.4f}s")

    # Calculate speedup
    speedup = (first_run_times[0] - cached_time) / first_run_times[0]
    print(f"Speedup from caching: {speedup:.1%} faster")

    # Sequential execution demonstration
    print("\nUsing execution_options to control execution:")
    with execution_options(scheduler="sequential"):
        start_time = time.perf_counter()
        result = pipeline(inputs={"value": 20})
        sequential_time = time.perf_counter() - start_time

        print(f"Result: {result}")
        print(f"Value: {result['value']}")
        print("Expected calculation: 20 + 10 = 30, then × 3 = 90")
        print(f"Time: {sequential_time:.4f}s (sequential execution)")
        
    # Get JIT statistics
    stats = get_jit_stats(pipeline)
    print("\nJIT Statistics:")
    print(f"Cache hits: {stats.get('cache_hits', 0)}")
    print(f"Cache misses: {stats.get('cache_misses', 0)}")
    print(f"Strategy: {stats.get('strategy', 'unknown')}")

    # Add a summary
    print("\n=== Summary ===")
    print(f"First run time: {first_run_times[0]:.4f}s (includes tracing overhead)")
    print(f"Cached run time: {cached_time:.4f}s")
    print(f"Sequential execution time: {sequential_time:.4f}s")
    print("\nKey benefits of autograph with JIT:")
    print("1. Automatic operator dependency discovery")
    print("2. Optimized execution with caching")
    print("3. Flexible execution strategies (parallel, sequential)")
    print("4. No manual graph construction required")


if __name__ == "__main__":
    main()
