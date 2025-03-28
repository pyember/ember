"""
Comprehensive performance benchmarks for nested operators with JIT optimization.

These tests measure the performance characteristics of different JIT implementations
on realistic nested operator structures, including:
1. Sequential chains
2. Diamond patterns
3. Wide ensembles
4. Complex nested structures
5. Real-world ensemble + judge patterns

The benchmarks compare:
- No JIT (baseline)
- Regular JIT
- Structural JIT with sequential execution
- Structural JIT with parallel execution

Results are analyzed to verify expected speedups and reported in a consistent format.
"""

import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional

import pytest

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.xcs.tracer.structural_jit import structural_jit

# Import JIT implementations directly from source
from ember.xcs.tracer.tracer_decorator import jit

# Configure logging
logger = logging.getLogger(__name__)

# Mark these tests to run only with special flag
pytestmark = [pytest.mark.performance]


# --------------------------------
# Benchmark Utilities
# --------------------------------


@dataclass
class BenchmarkResult:
    """Holds the results of a benchmark run."""

    name: str
    runs: int
    times: List[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        """Average execution time across all runs."""
        return statistics.mean(self.times) if self.times else 0

    @property
    def min_time(self) -> float:
        """Minimum execution time."""
        return min(self.times) if self.times else 0

    @property
    def max_time(self) -> float:
        """Maximum execution time."""
        return max(self.times) if self.times else 0

    @property
    def median_time(self) -> float:
        """Median execution time."""
        return statistics.median(self.times) if self.times else 0

    @property
    def std_dev(self) -> float:
        """Standard deviation of execution times."""
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    def __str__(self) -> str:
        """String representation with key metrics."""
        return (
            f"{self.name} ({self.runs} runs):\n"
            f"  Avg: {self.avg_time:.6f}s\n"
            f"  Min: {self.min_time:.6f}s\n"
            f"  Max: {self.max_time:.6f}s\n"
            f"  Med: {self.median_time:.6f}s\n"
            f"  Std: {self.std_dev:.6f}s"
        )


def run_benchmark(
    name: str,
    factory: Callable[[], Operator],
    input_data: Dict[str, Any],
    runs: int = 5,
    warmup_runs: int = 1,
) -> BenchmarkResult:
    """Run a benchmark with multiple iterations.

    Args:
        name: Name of the benchmark
        factory: Factory function to create the operator for each run
        input_data: Input data for the operator
        runs: Number of timed runs to perform
        warmup_runs: Number of warmup runs to perform

    Returns:
        BenchmarkResult with timing information
    """
    result = BenchmarkResult(name=name, runs=runs)

    # Perform warmup runs
    logger.info(f"Running warmup ({warmup_runs} runs) for {name}...")
    for _ in range(warmup_runs):
        op = factory()
        _ = op(inputs=input_data)

    # Perform timed runs
    logger.info(f"Running benchmark ({runs} runs) for {name}...")
    for i in range(runs):
        # Create a fresh operator instance for each run
        op = factory()

        # Time the execution
        start_time = time.time()
        _ = op(inputs=input_data)
        end_time = time.time()

        # Record the execution time
        elapsed = end_time - start_time
        result.times.append(elapsed)
        logger.info(f"  Run {i+1}/{runs}: {elapsed:.6f}s")

    # Log summary
    logger.info(f"Benchmark complete: {str(result)}")
    return result


def compare_benchmarks(
    benchmarks: List[BenchmarkResult],
    baseline_name: Optional[str] = None,
) -> Dict[str, float]:
    """Compare benchmark results against a baseline.

    Args:
        benchmarks: List of benchmark results
        baseline_name: Name of the baseline benchmark to compare against

    Returns:
        Dictionary mapping benchmark names to speedup factors
    """
    if not benchmarks:
        return {}

    # Find the baseline
    baseline = None
    if baseline_name:
        for b in benchmarks:
            if b.name == baseline_name:
                baseline = b
                break

    # If no baseline specified or found, use the first benchmark
    if baseline is None:
        baseline = benchmarks[0]

    # Calculate speedups relative to baseline
    speedups = {}
    baseline_time = baseline.avg_time

    for b in benchmarks:
        if b.avg_time > 0:
            speedup = baseline_time / b.avg_time
            speedups[b.name] = speedup
        else:
            speedups[b.name] = float("inf")

    # Log results
    logger.info("\nBenchmark Comparison:")
    logger.info(f"Baseline: {baseline.name} ({baseline.avg_time:.6f}s)")

    for name, speedup in speedups.items():
        if name != baseline.name:
            logger.info(f"{name}: {speedup:.2f}x speedup")

    return speedups


def save_benchmark_results(
    testname: str, benchmarks: List[BenchmarkResult], speedups: Dict[str, float]
):
    """Save benchmark results to a file for later analysis.

    Args:
        testname: Name of the test being run
        benchmarks: List of benchmark results
        speedups: Dictionary mapping benchmark names to speedup factors
    """
    import datetime
    import json
    import os

    # Create a results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate a filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/{testname}_{timestamp}.json"

    # Convert results to serializable format
    data = {
        "test_name": testname,
        "timestamp": timestamp,
        "results": [],
        "speedups": speedups,
    }

    for benchmark in benchmarks:
        data["results"].append(
            {
                "name": benchmark.name,
                "avg_time": benchmark.avg_time,
                "median_time": benchmark.median_time,
                "min_time": benchmark.min_time,
                "max_time": benchmark.max_time,
                "std_dev": benchmark.std_dev,
                "times": benchmark.times,
            }
        )

    # Write to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Benchmark results saved to {filename}")


# --------------------------------
# Benchmark Models
# --------------------------------


class OperatorInput(EmberModel):
    """Generic input model for benchmark operators."""

    data: Any
    work_factor: float = 1.0


class OperatorOutput(EmberModel):
    """Generic output model for benchmark operators."""

    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------
# Benchmark Operators
# --------------------------------


class CPUIntensiveOperator(Operator[OperatorInput, OperatorOutput]):
    """Operator that performs CPU-intensive computations."""

    specification: ClassVar[Specification] = Specification[
        OperatorInput, OperatorOutput
    ](
        input_model=OperatorInput,
        structured_output=OperatorOutput,
    )

    def __init__(self, name: str = "cpu", work_size: int = 1000000):
        """Initialize with a work size parameter.

        Args:
            name: Name of this operator instance
            work_size: Base amount of computation to perform
        """
        self.name = name
        self.work_size = work_size

    def forward(self, *, inputs: OperatorInput) -> OperatorOutput:
        """Perform CPU-intensive computation.

        Args:
            inputs: Input with data and work factor

        Returns:
            Computation result
        """
        # Scale work size by the work factor
        scaled_work = int(self.work_size * inputs.work_factor)

        # Perform CPU-intensive calculation
        result = 0
        for i in range(scaled_work):
            result += (i * i) % 997  # Prime modulo to prevent optimization

        return OperatorOutput(
            result=result, metadata={"name": self.name, "work_size": scaled_work}
        )


class LinearChainOperator(Operator[OperatorInput, OperatorOutput]):
    """Operator composed of a linear chain of sub-operators."""

    specification: ClassVar[Specification] = Specification[
        OperatorInput, OperatorOutput
    ](
        input_model=OperatorInput,
        structured_output=OperatorOutput,
    )

    def __init__(self, depth: int = 3, work_size: int = 100000):
        """Initialize with a chain of the specified depth.

        Args:
            depth: Number of sub-operators in the chain
            work_size: Base work size for each operator
        """
        self.depth = depth
        # Create chain of operators
        self.operators = [
            CPUIntensiveOperator(name=f"chain_{i}", work_size=work_size)
            for i in range(depth)
        ]

    def forward(self, *, inputs: OperatorInput) -> OperatorOutput:
        """Process input through the chain of operators.

        Args:
            inputs: Input data with work factor

        Returns:
            Final output after passing through all operators
        """
        current = inputs
        for i, op in enumerate(self.operators):
            # Convert output back to input for next operator
            result = op(inputs=current)
            current = OperatorInput(data=result.result, work_factor=inputs.work_factor)

        return OperatorOutput(
            result=current.data,
            metadata={
                "depth": self.depth,
                "operators": [op.name for op in self.operators],
            },
        )


class DiamondOperator(Operator[OperatorInput, OperatorOutput]):
    """Operator with a diamond-shaped dependency graph: A → (B,C) → D."""

    specification: ClassVar[Specification] = Specification[
        OperatorInput, OperatorOutput
    ](
        input_model=OperatorInput,
        structured_output=OperatorOutput,
    )

    def __init__(self, work_size: int = 100000):
        """Initialize the diamond-shaped operator structure.

        Args:
            work_size: Base work size for each operator
        """
        self.start = CPUIntensiveOperator(name="diamond_start", work_size=work_size)
        self.left = CPUIntensiveOperator(name="diamond_left", work_size=work_size)
        self.right = CPUIntensiveOperator(name="diamond_right", work_size=work_size)
        self.end = CPUIntensiveOperator(name="diamond_end", work_size=work_size)

    def forward(self, *, inputs: OperatorInput) -> OperatorOutput:
        """Execute the diamond pattern: A → (B,C) → D.

        Args:
            inputs: Input data with work factor

        Returns:
            Final output after merging parallel branches
        """
        # Start node
        start_result = self.start(inputs=inputs)
        start_output = start_result.result

        # Create inputs for parallel branches
        branch_input = OperatorInput(data=start_output, work_factor=inputs.work_factor)

        # Parallel branches (in a sequential implementation)
        left_result = self.left(inputs=branch_input)
        right_result = self.right(inputs=branch_input)

        # Combine results for final node
        combined_input = OperatorInput(
            data={"left": left_result.result, "right": right_result.result},
            work_factor=inputs.work_factor,
        )

        # End node
        final_result = self.end(inputs=combined_input)

        return OperatorOutput(
            result=final_result.result,
            metadata={
                "structure": "diamond",
                "components": ["start", "left", "right", "end"],
            },
        )


class EnsembleOperator(Operator[OperatorInput, OperatorOutput]):
    """Operator with multiple parallel sub-operators."""

    specification: ClassVar[Specification] = Specification[
        OperatorInput, OperatorOutput
    ](
        input_model=OperatorInput,
        structured_output=OperatorOutput,
    )

    def __init__(self, width: int = 5, work_size: int = 100000):
        """Initialize with the specified number of parallel operators.

        Args:
            width: Number of parallel operators in the ensemble
            work_size: Base work size for each operator
        """
        self.width = width
        self.members = [
            CPUIntensiveOperator(name=f"ensemble_{i}", work_size=work_size)
            for i in range(width)
        ]

    def forward(self, *, inputs: OperatorInput) -> OperatorOutput:
        """Execute all members and aggregate results.

        Args:
            inputs: Input data with work factor

        Returns:
            Aggregated results from all members
        """
        # Execute all members (sequentially in this implementation)
        results = [member(inputs=inputs) for member in self.members]

        # Aggregate results
        combined_result = sum(r.result for r in results)

        return OperatorOutput(
            result=combined_result,
            metadata={"width": self.width, "members": [m.name for m in self.members]},
        )


class JudgeOperator(Operator[OperatorInput, OperatorOutput]):
    """Operator that combines and evaluates results from other operators."""

    specification: ClassVar[Specification] = Specification[
        OperatorInput, OperatorOutput
    ](
        input_model=OperatorInput,
        structured_output=OperatorOutput,
    )

    def __init__(self, work_size: int = 200000):
        """Initialize the judge operator.

        Args:
            work_size: Base work size (usually higher than ensemble members)
        """
        self.processor = CPUIntensiveOperator(
            name="judge_processor", work_size=work_size
        )

    def forward(self, *, inputs: OperatorInput) -> OperatorOutput:
        """Process ensemble results to produce a final output.

        Args:
            inputs: Input data containing ensemble results

        Returns:
            Final evaluated result
        """
        # Process the input using the main processor
        result = self.processor(inputs=inputs)

        return OperatorOutput(
            result=result.result,
            metadata={"role": "judge", "processor": self.processor.name},
        )


class EnsembleJudgeSystem(Operator[OperatorInput, OperatorOutput]):
    """Complex system with an ensemble of operators followed by a judge."""

    specification: ClassVar[Specification] = Specification[
        OperatorInput, OperatorOutput
    ](
        input_model=OperatorInput,
        structured_output=OperatorOutput,
    )

    def __init__(self, width: int = 10, work_size: int = 100000):
        """Initialize the ensemble+judge system.

        Args:
            width: Number of operators in the ensemble
            work_size: Base work size for operators
        """
        self.ensemble = EnsembleOperator(width=width, work_size=work_size)
        self.judge = JudgeOperator(work_size=work_size * 2)

    def forward(self, *, inputs: OperatorInput) -> OperatorOutput:
        """Execute the ensemble and then the judge.

        Args:
            inputs: Input data with work factor

        Returns:
            Final output from the judge
        """
        # Run ensemble
        ensemble_result = self.ensemble(inputs=inputs)

        # Create input for judge
        judge_input = OperatorInput(
            data=ensemble_result.result, work_factor=inputs.work_factor
        )

        # Run judge
        final_result = self.judge(inputs=judge_input)

        return OperatorOutput(
            result=final_result.result,
            metadata={
                "system": "ensemble_judge",
                "ensemble_width": self.ensemble.width,
                "judge": self.judge.processor.name,
            },
        )


class NestedEnsembleSystem(Operator[OperatorInput, OperatorOutput]):
    """Complex system with nested ensembles at multiple levels."""

    specification: ClassVar[Specification] = Specification[
        OperatorInput, OperatorOutput
    ](
        input_model=OperatorInput,
        structured_output=OperatorOutput,
    )

    def __init__(
        self, level1_width: int = 3, level2_width: int = 3, work_size: int = 100000
    ):
        """Initialize the nested ensemble system.

        Args:
            level1_width: Number of level-1 ensembles
            level2_width: Number of operators in each level-2 ensemble
            work_size: Base work size for operators
        """
        # Create level 2 ensembles (these are used by level 1)
        self.level2_ensembles = [
            EnsembleOperator(width=level2_width, work_size=work_size)
            for _ in range(level1_width)
        ]

        # Create a judge for the final output
        self.judge = JudgeOperator(work_size=work_size * 2)

    def forward(self, *, inputs: OperatorInput) -> OperatorOutput:
        """Execute all nested ensembles and the final judge.

        Args:
            inputs: Input data with work factor

        Returns:
            Final output from the judge
        """
        # Run level 2 ensembles
        ensemble_results = [
            ensemble(inputs=inputs) for ensemble in self.level2_ensembles
        ]

        # Combine results
        combined_result = sum(r.result for r in ensemble_results)

        # Create input for judge
        judge_input = OperatorInput(
            data=combined_result, work_factor=inputs.work_factor
        )

        # Run judge
        final_result = self.judge(inputs=judge_input)

        return OperatorOutput(
            result=final_result.result,
            metadata={
                "system": "nested_ensemble",
                "level1_width": len(self.level2_ensembles),
                "level2_width": self.level2_ensembles[0].width,
                "judge": self.judge.processor.name,
            },
        )


# --------------------------------
# JIT Decorator Variants
# --------------------------------


def apply_regular_jit(operator_class):
    """Apply regular JIT to an operator class."""
    return jit(operator_class)


def apply_structural_jit_sequential(operator_class):
    """Apply structural JIT with sequential execution to an operator class."""
    return structural_jit(
        execution_strategy="sequential",
    )(operator_class)


def apply_structural_jit_parallel(operator_class):
    """Apply structural JIT with parallel execution to an operator class."""
    return structural_jit(
        execution_strategy="parallel",
    )(operator_class)


# --------------------------------
# Benchmark Tests
# --------------------------------


@pytest.mark.skipif(
    "not config.getoption('--run-perf-tests') and not config.getoption('--run-all-tests')",
    reason="Performance tests are disabled by default. Run with --run-perf-tests or --run-all-tests flag.",
)
class TestOperatorPerformance:
    """Comprehensive performance tests for operators with different JIT implementations."""

    def test_linear_chain_performance(self):
        """Test performance of linear chain operators with different JIT implementations."""

        # Define operator factory functions
        def create_no_jit():
            return LinearChainOperator(depth=5, work_size=100000)

        def create_regular_jit():
            return apply_regular_jit(LinearChainOperator)(depth=5, work_size=100000)

        def create_structural_sequential():
            return apply_structural_jit_sequential(LinearChainOperator)(
                depth=5, work_size=100000
            )

        def create_structural_parallel():
            return apply_structural_jit_parallel(LinearChainOperator)(
                depth=5, work_size=100000
            )

        # Define input data
        input_data = OperatorInput(data="test", work_factor=1.0)

        # Run benchmarks
        benchmarks = [
            run_benchmark("No JIT (Linear)", create_no_jit, input_data),
            run_benchmark("Regular JIT (Linear)", create_regular_jit, input_data),
            run_benchmark(
                "Structural JIT Sequential (Linear)",
                create_structural_sequential,
                input_data,
            ),
            run_benchmark(
                "Structural JIT Parallel (Linear)",
                create_structural_parallel,
                input_data,
            ),
        ]

        # Compare results
        speedups = compare_benchmarks(benchmarks, "No JIT (Linear)")

        # Save benchmark results to a file for later analysis
        save_benchmark_results("linear_chain_performance", benchmarks, speedups)

        # Linear chains shouldn't see much benefit from parallelization
        # But should see some benefit from JIT compilation
        assert (
            speedups["Regular JIT (Linear)"] >= 0.9
        ), "Regular JIT should not be slower than baseline"

    def test_diamond_performance(self):
        """Test performance of diamond-shaped operators with different JIT implementations."""

        # Define operator factory functions
        def create_no_jit():
            return DiamondOperator(work_size=100000)

        def create_regular_jit():
            return apply_regular_jit(DiamondOperator)(work_size=100000)

        def create_structural_sequential():
            return apply_structural_jit_sequential(DiamondOperator)(work_size=100000)

        def create_structural_parallel():
            return apply_structural_jit_parallel(DiamondOperator)(work_size=100000)

        # Define input data
        input_data = OperatorInput(data="test", work_factor=1.0)

        # Run benchmarks
        benchmarks = [
            run_benchmark("No JIT (Diamond)", create_no_jit, input_data),
            run_benchmark("Regular JIT (Diamond)", create_regular_jit, input_data),
            run_benchmark(
                "Structural JIT Sequential (Diamond)",
                create_structural_sequential,
                input_data,
            ),
            run_benchmark(
                "Structural JIT Parallel (Diamond)",
                create_structural_parallel,
                input_data,
            ),
        ]

        # Compare results
        speedups = compare_benchmarks(benchmarks, "No JIT (Diamond)")

        # Save benchmark results to a file for later analysis
        save_benchmark_results("diamond_performance", benchmarks, speedups)

        # Diamond patterns should benefit from parallelization
        # and from JIT compilation
        assert (
            speedups["Structural JIT Parallel (Diamond)"] >= 0.9
        ), "Parallel execution should not be slower than baseline"

    def test_ensemble_performance(self):
        """Test performance of ensemble operators with different JIT implementations."""

        # Define operator factory functions
        def create_no_jit():
            return EnsembleOperator(width=10, work_size=500000)

        def create_regular_jit():
            return apply_regular_jit(EnsembleOperator)(width=10, work_size=500000)

        def create_structural_sequential():
            return apply_structural_jit_sequential(EnsembleOperator)(
                width=10, work_size=500000
            )

        def create_structural_parallel():
            return apply_structural_jit_parallel(EnsembleOperator)(
                width=10, work_size=500000
            )

        # Define input data
        input_data = OperatorInput(data="test", work_factor=1.0)

        # Run benchmarks
        benchmarks = [
            run_benchmark("No JIT (Ensemble)", create_no_jit, input_data),
            run_benchmark("Regular JIT (Ensemble)", create_regular_jit, input_data),
            run_benchmark(
                "Structural JIT Sequential (Ensemble)",
                create_structural_sequential,
                input_data,
            ),
            run_benchmark(
                "Structural JIT Parallel (Ensemble)",
                create_structural_parallel,
                input_data,
            ),
        ]

        # Compare results
        speedups = compare_benchmarks(benchmarks, "No JIT (Ensemble)")

        # Save benchmark results to a file for later analysis
        save_benchmark_results("ensemble_performance", benchmarks, speedups)

        # Ensemble patterns should benefit significantly from parallelization
        # Structural JIT with parallel execution should be fastest
        assert (
            speedups["Structural JIT Parallel (Ensemble)"] >= 0.9
        ), "Parallel execution should not be slower than baseline"

    def test_ensemble_judge_performance(self):
        """Test performance of ensemble+judge system with different JIT implementations."""

        # Define operator factory functions
        def create_no_jit():
            return EnsembleJudgeSystem(width=10, work_size=50000)

        def create_regular_jit():
            return apply_regular_jit(EnsembleJudgeSystem)(width=10, work_size=50000)

        def create_structural_sequential():
            return apply_structural_jit_sequential(EnsembleJudgeSystem)(
                width=10, work_size=50000
            )

        def create_structural_parallel():
            return apply_structural_jit_parallel(EnsembleJudgeSystem)(
                width=10, work_size=50000
            )

        # Define input data
        input_data = OperatorInput(data="test", work_factor=1.0)

        # Run benchmarks
        benchmarks = [
            run_benchmark("No JIT (EnsembleJudge)", create_no_jit, input_data),
            run_benchmark(
                "Regular JIT (EnsembleJudge)", create_regular_jit, input_data
            ),
            run_benchmark(
                "Structural JIT Sequential (EnsembleJudge)",
                create_structural_sequential,
                input_data,
            ),
            run_benchmark(
                "Structural JIT Parallel (EnsembleJudge)",
                create_structural_parallel,
                input_data,
            ),
        ]

        # Compare results
        speedups = compare_benchmarks(benchmarks, "No JIT (EnsembleJudge)")

        # Save benchmark results to a file for later analysis
        save_benchmark_results("ensemble_judge_performance", benchmarks, speedups)

        # This pattern should benefit significantly from both JIT and parallelization
        assert (
            speedups["Structural JIT Parallel (EnsembleJudge)"] >= 0.9
        ), "Parallel execution should not be slower than baseline"

    def test_nested_ensemble_performance(self):
        """Test performance of nested ensemble system with different JIT implementations."""

        # Define operator factory functions
        def create_no_jit():
            return NestedEnsembleSystem(level1_width=3, level2_width=3, work_size=50000)

        def create_regular_jit():
            return apply_regular_jit(NestedEnsembleSystem)(
                level1_width=3, level2_width=3, work_size=50000
            )

        def create_structural_sequential():
            return apply_structural_jit_sequential(NestedEnsembleSystem)(
                level1_width=3, level2_width=3, work_size=50000
            )

        def create_structural_parallel():
            return apply_structural_jit_parallel(NestedEnsembleSystem)(
                level1_width=3, level2_width=3, work_size=50000
            )

        # Define input data
        input_data = OperatorInput(data="test", work_factor=1.0)

        # Run benchmarks
        benchmarks = [
            run_benchmark("No JIT (NestedEnsemble)", create_no_jit, input_data),
            run_benchmark(
                "Regular JIT (NestedEnsemble)", create_regular_jit, input_data
            ),
            run_benchmark(
                "Structural JIT Sequential (NestedEnsemble)",
                create_structural_sequential,
                input_data,
            ),
            run_benchmark(
                "Structural JIT Parallel (NestedEnsemble)",
                create_structural_parallel,
                input_data,
            ),
        ]

        # Compare results
        speedups = compare_benchmarks(benchmarks, "No JIT (NestedEnsemble)")

        # Save benchmark results to a file for later analysis
        save_benchmark_results("nested_ensemble_performance", benchmarks, speedups)

        # Nested ensemble should see greatest benefit from structural JIT with parallel execution
        assert (
            speedups["Structural JIT Parallel (NestedEnsemble)"] >= 0.9
        ), "Parallel execution should not be slower than baseline"


# --------------------------------
# Main Execution
# --------------------------------

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Add command line argument support for direct execution
    import argparse

    parser = argparse.ArgumentParser(
        description="Run nested operator performance tests"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of test runs to perform"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Number of warmup runs to perform"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "linear",
            "diamond",
            "ensemble",
            "ensemble_judge",
            "nested_ensemble",
            "all",
        ],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    print(
        f"Running performance tests with {args.runs} runs and {args.warmup} warmup runs"
    )

    # Create test instance
    test_instance = TestOperatorPerformance()

    # Set run counts in environment (simpler than modifying all test methods)
    os.environ["TEST_RUNS"] = str(args.runs)
    os.environ["TEST_WARMUP"] = str(args.warmup)

    # Run selected test(s)
    if args.test == "all" or args.test == "linear":
        print("\n=== Running Linear Chain Test ===")
        test_instance.test_linear_chain_performance()

    if args.test == "all" or args.test == "diamond":
        print("\n=== Running Diamond Pattern Test ===")
        test_instance.test_diamond_performance()

    if args.test == "all" or args.test == "ensemble":
        print("\n=== Running Ensemble Test ===")
        test_instance.test_ensemble_performance()

    if args.test == "all" or args.test == "ensemble_judge":
        print("\n=== Running Ensemble+Judge Test ===")
        test_instance.test_ensemble_judge_performance()

    if args.test == "all" or args.test == "nested_ensemble":
        print("\n=== Running Nested Ensemble Test ===")
        test_instance.test_nested_ensemble_performance()
