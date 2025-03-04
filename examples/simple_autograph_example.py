"""Simple Auto Graph Example with Mock Operators.

This example demonstrates the enhanced JIT API with automatic graph building
using simple mock operators that don't require external dependencies.
"""

import logging
import time
from typing import Any, Dict, List

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.prompt_specification.specification import Specification
from ember.xcs.tracer.tracer_decorator import jit
from ember.xcs.engine.execution_options import execution_options


###############################################################################
# Mock Operators
###############################################################################


@jit()
class AddOperator(Operator):
    """Simple operator that adds a value to the input."""

    specification = Specification(input_model=None, output_model=None)

    def __init__(self, *, value: int = 1) -> None:
        self.value = value
        self.specification = Specification(input_model=None, output_model=None)

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs.get("value", 0) + self.value
        return {"value": result}


@jit()
class MultiplyOperator(Operator):
    """Simple operator that multiplies the input by a value."""

    specification = Specification(input_model=None, output_model=None)

    def __init__(self, *, value: int = 2) -> None:
        self.value = value
        self.specification = Specification(input_model=None, output_model=None)

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs.get("value", 0) * self.value
        return {"value": result}


@jit()
class DelayOperator(Operator):
    """Simple operator that introduces a delay."""

    specification = Specification(input_model=None, output_model=None)

    def __init__(self, *, delay: float = 0.1) -> None:
        self.delay = delay
        self.specification = Specification(input_model=None, output_model=None)

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

    specification = Specification(input_model=None, output_model=None)

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
    for i, inp in enumerate(inputs):
        print(f"\nInput {i+1}: {inp}")

        start_time = time.perf_counter()
        result = pipeline(inputs=inp)
        elapsed = time.perf_counter() - start_time

        print(f"Result: {result}")
        print(f"Time: {elapsed:.4f}s")

    print("\nRepeat first input to demonstrate cached execution:")
    start_time = time.perf_counter()
    result = pipeline(inputs=inputs[0])
    elapsed = time.perf_counter() - start_time

    print(f"Result: {result}")
    print(f"Time: {elapsed:.4f}s")

    print("\nUsing execution_options to control execution:")
    with execution_options(scheduler="sequential"):
        start_time = time.perf_counter()
        result = pipeline(inputs={"value": 20})
        elapsed = time.perf_counter() - start_time

        print(f"Result: {result}")
        print(f"Time: {elapsed:.4f}s (sequential execution)")


if __name__ == "__main__":
    main()
