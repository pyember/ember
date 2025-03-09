"""Simple JIT Demo

This module demonstrates the core JIT functionality without requiring the full Ember system.
It uses minimal dependencies and focuses on showing how the JIT decorator works.

To run:
    poetry run python src/ember/examples/simple_jit_demo.py
"""

import time
from typing import Any, Dict, List, Type

# Import the JIT decorator from the API
from ember.api.xcs import jit
from ember.api.operator import Operator, Specification
from ember.api.types import EmberModel


# Define input/output models
class DemoInput(EmberModel):
    query: str

class DemoOutput(EmberModel):
    result: str

# Define specifications
class DemoSpecification(Specification):
    input_model: Type[EmberModel] = DemoInput
    output_model: Type[EmberModel] = DemoOutput

# Define a simple operator that introduces a delay
class DelayOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Operator that sleeps for a specified time and returns a result."""
    
    specification = DemoSpecification()

    def __init__(self, *, delay_seconds: float) -> None:
        """Initialize with delay time in seconds."""
        self.delay_seconds = delay_seconds

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with a delay and return result."""
        time.sleep(self.delay_seconds)
        return {"result": f"Completed after {self.delay_seconds}s delay"}


# Define CompositeOperator input/output models
class CompositeInput(EmberModel):
    query: str

class CompositeOutput(EmberModel):
    results: List[Dict[str, Any]]

# Define CompositeOperator specification
class CompositeSpecification(Specification):
    input_model: Type[EmberModel] = CompositeInput
    output_model: Type[EmberModel] = CompositeOutput

# Apply JIT to a composite operator
@jit(sample_input={"query": "test"})
class CompositeOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Operator that composes multiple DelayOperators."""
    
    specification = CompositeSpecification()

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
    result1 = jit_op(inputs={"query": "first run"})
    first_time = time.time() - start
    print(f"  Input: 'first run'")
    print(f"  Results: {len(result1.results)} operations completed")
    for i, res in enumerate(result1.results):
        print(f"    - Op {i+1}: {res['result']}")
    print(f"  Total time: {first_time:.4f}s")

    print("\nSecond execution (JIT should use cached plan):")
    start = time.time()
    result2 = jit_op(inputs={"query": "second run"})
    second_time = time.time() - start
    print(f"  Input: 'second run'")
    print(f"  Results: {len(result2.results)} operations completed")
    print(f"  Total time: {second_time:.4f}s")
    print(f"  Speed improvement: {(first_time - second_time) / first_time:.1%} faster than first run")

    print("\nStandard execution (for comparison):")
    start = time.time()
    result3 = standard_op(inputs={"query": "comparison"})
    standard_time = time.time() - start
    print(f"  Input: 'comparison'")
    print(f"  Results: {len(result3.results)} operations completed")
    print(f"  Total time: {standard_time:.4f}s")

    print("\nPerformance Summary:")
    print(f"  First run (with tracing): {first_time:.4f}s")
    print(f"  Second run (cached): {second_time:.4f}s")
    print(f"  Standard execution: {standard_time:.4f}s")
    
    # Calculate performance improvements
    jit_vs_standard = (standard_time - second_time) / standard_time
    trace_overhead = (first_time - standard_time) / standard_time
    
    print("\nOptimization Analysis:")
    if second_time < standard_time:
        print(f"  ✅ Success! JIT execution is {jit_vs_standard:.1%} faster than standard execution.")
    else:
        print(f"  ⚠️ JIT execution is {-jit_vs_standard:.1%} slower than standard execution.")
        print("     This might be due to test conditions or small operation count.")
    
    print(f"  Tracing overhead: {trace_overhead:.1%} compared to standard execution")
    print(f"  Time saved by JIT on second run: {first_time - second_time:.4f}s")
    
    print("\nWhen to use JIT:")
    print("  - Complex computation graphs with many operations")
    print("  - Operations that are executed multiple times with similar inputs")
    print("  - When parallel execution can help performance")


if __name__ == "__main__":
    main()
