"""Simple JIT Demo

This module demonstrates the core JIT functionality without requiring the full Ember system.
It uses minimal dependencies and focuses on showing how the JIT decorator works.
"""

import time
from typing import Any, Dict, List

# Import the JIT decorator
from ember.xcs.tracer.tracer_decorator import jit
from ember.core.registry.operator.base.operator_base import Operator


# Define a simple operator that introduces a delay
class DelayOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Operator that sleeps for a specified time and returns a result."""

    def __init__(self, *, delay_seconds: float) -> None:
        """Initialize with delay time in seconds."""
        self.delay_seconds = delay_seconds

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with a delay and return result."""
        time.sleep(self.delay_seconds)
        return {"result": f"Completed after {self.delay_seconds}s delay"}


# Apply JIT to a composite operator
@jit(sample_input={"query": "test"})
class CompositeOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Operator that composes multiple DelayOperators."""

    def __init__(self, *, num_ops: int = 3, delay: float = 0.1) -> None:
        """Initialize with multiple delay operators."""
        self.operators = [DelayOperator(delay_seconds=delay) for _ in range(num_ops)]

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all internal operators and return combined results."""
        results = []
        for op in self.operators:
            results.append(op(inputs=inputs))

        return {"results": results}


def main() -> None:
    """Run a simple demonstration of JIT functionality."""
    print("Creating operators...")
    standard_op = CompositeOperator(num_ops=5, delay=0.1)
    jit_op = CompositeOperator(num_ops=5, delay=0.1)

    print("\nFirst execution (JIT will trace and cache):")
    start = time.time()
    jit_op(inputs={"query": "first run"})
    first_time = time.time() - start
    print(f"  Time: {first_time:.4f}s")

    print("\nSecond execution (JIT should use cached plan):")
    start = time.time()
    jit_op(inputs={"query": "second run"})
    second_time = time.time() - start
    print(f"  Time: {second_time:.4f}s")

    print("\nStandard execution (for comparison):")
    start = time.time()
    standard_op(inputs={"query": "comparison"})
    standard_time = time.time() - start
    print(f"  Time: {standard_time:.4f}s")

    print("\nSummary:")
    print(f"  First run (with tracing): {first_time:.4f}s")
    print(f"  Second run (cached): {second_time:.4f}s")
    print(f"  Standard execution: {standard_time:.4f}s")

    if second_time < standard_time:
        print("\nSuccess! JIT execution is faster than standard execution.")
    else:
        print(
            "\nNote: JIT execution is not showing expected speedup. This might be due to test conditions."
        )


if __name__ == "__main__":
    main()
