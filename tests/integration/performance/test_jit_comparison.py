"""Performance comparison between regular JIT and structural JIT.

This module compares regular JIT (based on execution tracing) with structural JIT
(based on operator structure analysis) to determine which approach provides better
performance for different operator patterns.
"""

import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar, List

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field
from ember.xcs.tracer.structural_jit import structural_jit

# Import JIT implementations
from ember.xcs.tracer.tracer_decorator import jit

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------
# Models
# --------------------------------
class DelayInput(EmberModel):
    """Input model for delay-based operations."""

    task_id: str = Field(description="Unique identifier for the task")
    delay_factor: float = Field(
        default=1.0, description="Factor to scale the base delay"
    )


class DelayOutput(EmberModel):
    """Output model from delay-based operations."""

    result: str = Field(description="Operation result")
    task_id: str = Field(description="Task identifier")
    processing_time: float = Field(description="Time taken to process")


class EnsembleOutput(EmberModel):
    """Output model from ensemble operations."""

    results: List[str] = Field(description="Results from sub-operations")
    task_id: str = Field(description="Task identifier")
    execution_times: List[float] = Field(description="Times for each sub-operation")


# --------------------------------
# Base Operators - Properly Structured
# --------------------------------
class DelayOperator(Operator[DelayInput, DelayOutput]):
    """Operator that introduces a fixed delay to simulate processing.

    Properly structured with class variables and initialization to enable
    both regular JIT and structural JIT to analyze it correctly.
    """

    # Class-level specification
    specification: ClassVar[Specification] = Specification(
        input_model=DelayInput,
        structured_output=DelayOutput,
    )

    # Field declarations - important for structural JIT
    name: str
    delay: float

    def __init__(self, *, name: str, delay: float) -> None:
        """Initialize with configuration parameters.

        Args:
            name: Identifier for this operator
            delay: Base delay in seconds
        """
        self.name = name
        self.delay = delay

    def forward(self, *, inputs: DelayInput) -> DelayOutput:
        """Process input with a simulated delay.

        Args:
            inputs: Input parameters including task ID and delay factor

        Returns:
            Output with result and timing information
        """
        # Calculate actual delay using input factor
        actual_delay = self.delay * inputs.delay_factor

        # Record start time
        start_time = time.time()

        # Simulate processing with sleep
        time.sleep(actual_delay)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create and return result
        return DelayOutput(
            result=f"Processed by {self.name} in {processing_time:.4f}s",
            task_id=inputs.task_id,
            processing_time=processing_time,
        )


# --------------------------------
# Ensemble Operators with Different JIT Optimizations
# --------------------------------


# No JIT - baseline
class EnsembleOperator(Operator[DelayInput, EnsembleOutput]):
    """Ensemble operator that runs multiple delay operators sequentially.

    No JIT optimization, serves as a baseline for comparison.
    """

    # Class-level specification
    specification: ClassVar[Specification] = Specification(
        input_model=DelayInput,
        structured_output=EnsembleOutput,
    )

    # Field declarations
    members: List[DelayOperator]

    def __init__(self, *, width: int = 10, delay: float = 0.1) -> None:
        """Initialize with multiple delay operators.

        Args:
            width: Number of member operators
            delay: Delay for each operator in seconds
        """
        self.members = [
            DelayOperator(name=f"op_{i}", delay=delay) for i in range(width)
        ]

    def forward(self, *, inputs: DelayInput) -> EnsembleOutput:
        """Process input with all member operators sequentially.

        Args:
            inputs: Input parameters

        Returns:
            Collected results from all members
        """
        results = []
        execution_times = []

        # Sequential execution
        for i, member in enumerate(self.members):
            # Create subtask for this member
            subtask = DelayInput(
                task_id=f"{inputs.task_id}-{i}", delay_factor=inputs.delay_factor
            )

            # Execute member
            output = member(inputs=subtask)

            # Collect results
            results.append(output.result)
            execution_times.append(output.processing_time)

        return EnsembleOutput(
            results=results, task_id=inputs.task_id, execution_times=execution_times
        )


# Regular JIT - uses execution tracing
@jit
class RegularJITEnsembleOperator(EnsembleOperator):
    """Ensemble operator optimized with regular JIT.

    Uses the @jit decorator which is based on execution tracing.
    """

    pass


# Structural JIT - uses operator structure analysis
@structural_jit
class StructuralJITEnsembleOperator(EnsembleOperator):
    """Ensemble operator optimized with structural JIT.

    Uses the @structural_jit decorator which analyzes operator structure.
    """

    pass


# Structural JIT with explicit parallel execution
@structural_jit(execution_strategy="parallel")
class ParallelStructuralJITEnsembleOperator(EnsembleOperator):
    """Ensemble operator with structural JIT and explicit parallel execution.

    Uses structural JIT with a parallel execution strategy.
    """

    pass


# Explicit parallelism for reference
class ExplicitParallelEnsembleOperator(Operator[DelayInput, EnsembleOutput]):
    """Ensemble operator with explicit thread-based parallelism.

    Manually implements parallelism for comparison with JIT approaches.
    """

    # Class-level specification
    specification: ClassVar[Specification] = Specification(
        input_model=DelayInput,
        structured_output=EnsembleOutput,
    )

    # Field declarations
    members: List[DelayOperator]

    def __init__(self, *, width: int = 10, delay: float = 0.1) -> None:
        """Initialize with multiple delay operators.

        Args:
            width: Number of member operators
            delay: Delay for each operator in seconds
        """
        self.members = [
            DelayOperator(name=f"par_op_{i}", delay=delay) for i in range(width)
        ]

    def forward(self, *, inputs: DelayInput) -> EnsembleOutput:
        """Process input with all member operators in parallel.

        Args:
            inputs: Input parameters

        Returns:
            Collected results from all members
        """
        results = []
        execution_times = []

        # Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.members)) as executor:
            # Submit all tasks to the executor
            futures = []
            for i, member in enumerate(self.members):
                # Create subtask for this member
                subtask = DelayInput(
                    task_id=f"{inputs.task_id}-{i}", delay_factor=inputs.delay_factor
                )

                # Submit task to executor
                futures.append(executor.submit(member, inputs=subtask))

            # Collect results as they complete
            for future in as_completed(futures):
                output = future.result()
                results.append(output.result)
                execution_times.append(output.processing_time)

        return EnsembleOutput(
            results=results, task_id=inputs.task_id, execution_times=execution_times
        )


# --------------------------------
# Tests
# --------------------------------
def test_jit_comparison():
    """Compare performance of different JIT implementations.

    This test creates identical EnsembleOperators with different JIT decorators
    and compares their execution times.
    """
    # Configuration
    ensemble_width = 10
    delay = 0.1  # seconds per operation
    runs = 3  # number of test runs

    # Create operators with identical configuration
    baseline_op = EnsembleOperator(width=ensemble_width, delay=delay)
    regular_jit_op = RegularJITEnsembleOperator(width=ensemble_width, delay=delay)
    struct_jit_op = StructuralJITEnsembleOperator(width=ensemble_width, delay=delay)
    parallel_struct_jit_op = ParallelStructuralJITEnsembleOperator(
        width=ensemble_width, delay=delay
    )
    explicit_parallel_op = ExplicitParallelEnsembleOperator(
        width=ensemble_width, delay=delay
    )

    # Input data
    input_data = DelayInput(task_id="test_comparison", delay_factor=1.0)

    # Test functions to measure performance
    def measure_operator(op, name):
        logger.info(f"\n=== Testing {name} ===")
        times = []

        # First run (may include compilation/tracing overhead)
        start_time = time.time()
        result = op(inputs=input_data)
        first_time = time.time() - start_time
        logger.info(f"First run: {first_time:.4f}s")

        # Subsequent runs
        for i in range(runs):
            start_time = time.time()
            result = op(inputs=input_data)
            elapsed = time.time() - start_time
            times.append(elapsed)
            logger.info(f"Run {i+1}: {elapsed:.4f}s")

        # Calculate statistics
        avg_time = statistics.mean(times)
        logger.info(f"Average time: {avg_time:.4f}s")

        return {
            "name": name,
            "first_run": first_time,
            "times": times,
            "avg_time": avg_time,
        }

    # Measure all operators
    results = []
    results.append(measure_operator(baseline_op, "Baseline (No JIT)"))
    results.append(measure_operator(regular_jit_op, "Regular JIT"))
    results.append(measure_operator(struct_jit_op, "Structural JIT"))
    results.append(
        measure_operator(parallel_struct_jit_op, "Structural JIT (Parallel)")
    )
    results.append(measure_operator(explicit_parallel_op, "Explicit Parallel"))

    # Compare results
    baseline = next(r for r in results if r["name"] == "Baseline (No JIT)")
    baseline_time = baseline["avg_time"]

    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Baseline Time: {baseline_time:.4f}s")

    for result in results:
        if result["name"] != "Baseline (No JIT)":
            speedup = (
                baseline_time / result["avg_time"] if result["avg_time"] > 0 else 0
            )
            logger.info(
                f"{result['name']}: {result['avg_time']:.4f}s ({speedup:.2f}x speedup)"
            )

    # Get explicit parallel result for reference
    explicit = next(r for r in results if r["name"] == "Explicit Parallel")
    explicit_time = explicit["avg_time"]
    theoretical_speedup = (
        ensemble_width  # In theory, we could get ensemble_width times speedup
    )

    logger.info(f"\nExplicit Parallel Time: {explicit_time:.4f}s")
    logger.info(f"Theoretical speedup: {theoretical_speedup:.1f}x")

    # Check if structural JIT with parallel execution is approaching explicit parallel
    parallel_struct = next(
        r for r in results if r["name"] == "Structural JIT (Parallel)"
    )
    parallel_struct_time = parallel_struct["avg_time"]

    ratio_to_explicit = parallel_struct_time / explicit_time if explicit_time > 0 else 0
    logger.info(
        f"Structural JIT (Parallel) vs Explicit Parallel: {ratio_to_explicit:.2f}x"
    )

    # Verify at least some speedup from structural JIT with parallel execution
    regular_struct = next(r for r in results if r["name"] == "Structural JIT")
    regular_struct_time = regular_struct["avg_time"]

    parallel_vs_regular = (
        regular_struct_time / parallel_struct_time if parallel_struct_time > 0 else 0
    )
    logger.info(
        f"Structural JIT (Parallel) vs Regular Structural JIT: {parallel_vs_regular:.2f}x"
    )

    # Save results to file
    import datetime
    import json
    import os

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/jit_comparison_{timestamp}.json"

    # Save detailed results
    with open(filename, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "configuration": {
                    "ensemble_width": ensemble_width,
                    "delay": delay,
                    "runs": runs,
                },
                "results": results,
                "comparisons": {
                    "explicit_vs_baseline": baseline_time / explicit_time
                    if explicit_time > 0
                    else 0,
                    "parallel_struct_vs_baseline": baseline_time / parallel_struct_time
                    if parallel_struct_time > 0
                    else 0,
                    "parallel_struct_vs_explicit": ratio_to_explicit,
                    "parallel_vs_regular_struct": parallel_vs_regular,
                },
            },
            f,
            indent=2,
        )

    logger.info(f"\nResults saved to {filename}")


# Run the test if executed directly
if __name__ == "__main__":
    logger.info("=== JIT Implementation Comparison ===")
    test_jit_comparison()
