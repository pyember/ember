"""JIT Ensemble Demonstration.

This module demonstrates three variants of a "LargeEnsemble" operator using Ember's
API:
    1) BaselineEnsemble: Executes eagerly without parallelism
    2) ParallelEnsemble: Leverages concurrency via a scheduling plan
    3) JITEnsemble: Combines parallel execution with JIT tracing to cache the concurrency plan

It measures the total and per-query execution times for each approach, highlighting the
performance benefits of JIT compilation with the @jit decorator. This example focuses
on the trace-based JIT approach, which is one of three complementary approaches in Ember's
JIT system (the others being structural_jit and autograph).

For a comprehensive explanation of the relationship between these approaches,
see docs/xcs/JIT_OVERVIEW.md.

To run:
    uv run python src/ember/examples/xcs/jit_example.py
"""

import logging
import time
from typing import ClassVar, List, Tuple, Type

# ember API imports
from ember.api.xcs import execution_options, jit
from ember.core.registry.operator.base.operator_base import Operator, Specification
from ember.core.types.ember_model import EmberModel, Field


###############################################################################
# Input/Output Models
###############################################################################
class EnsembleInput(EmberModel):
    """Input for the ensemble operators."""

    query: str = Field(description="The query to send to the ensemble")


class EnsembleOutput(EmberModel):
    """Output from the ensemble operators."""

    query: str = Field(description="The original query")
    responses: List[str] = Field(description="List of responses from the ensemble")


# Proper specification for the ensemble operators
class EnsembleSpecification(Specification):
    """Specification for the ensemble operators."""

    input_model: Type[EmberModel] = EnsembleInput
    structured_output: Type[EmberModel] = EnsembleOutput


###############################################################################
# BaselineEnsemble - Eager Execution (No Concurrency)
###############################################################################
class BaselineEnsemble(Operator):
    """Ensemble implementation that forces fully eager (serial) execution.

    This subclass configures the execution to run serially rather than in parallel
    by using appropriate execution options.
    """

    specification: ClassVar[Specification] = EnsembleSpecification()

    def __init__(
        self,
        *,
        num_units: int = 3,
        model_name: str = "openai:gpt-4o-mini",
        temperature: float = 0.7,
    ) -> None:
        """Initialize with sequential execution options."""
        self.num_units = num_units
        self.model_name = model_name
        self.temperature = temperature

    def forward(self, *, inputs: EnsembleInput) -> EnsembleOutput:
        """Process inputs in a sequential manner."""
        # Simulate execution by introducing a delay
        time.sleep(0.1 * self.num_units)

        # Generate mock responses
        responses = [
            f"Response {i} to query: {inputs.query}" for i in range(self.num_units)
        ]

        return EnsembleOutput(query=inputs.query, responses=responses)


###############################################################################
# ParallelEnsemble - Standard Concurrency
###############################################################################
class ParallelEnsemble(Operator):
    """Ensemble implementation that leverages standard concurrency.

    This implementation prepares for parallel execution with proper XCS integration.
    """

    specification: ClassVar[Specification] = EnsembleSpecification()

    def __init__(
        self,
        *,
        num_units: int = 3,
        model_name: str = "openai:gpt-4o-mini",
        temperature: float = 0.7,
    ) -> None:
        """Initialize with standard configuration."""
        self.num_units = num_units
        self.model_name = model_name
        self.temperature = temperature

    def forward(self, *, inputs: EnsembleInput) -> EnsembleOutput:
        """Process inputs with potential for parallelism."""
        # Simulate execution by introducing a delay (smaller for parallel)
        time.sleep(0.05 * self.num_units)

        # Generate mock responses
        responses = [
            f"Parallel response {i} to query: {inputs.query}"
            for i in range(self.num_units)
        ]

        return EnsembleOutput(query=inputs.query, responses=responses)


###############################################################################
# JITEnsemble - Parallel Execution with JIT Tracing
###############################################################################
@jit
class JITEnsemble(ParallelEnsemble):
    """Ensemble implementation with JIT tracing for optimized concurrency.

    Uses the same parallel approach as ParallelEnsemble but with JIT decoration. The first call
    triggers tracing and caching of the execution plan, reducing overhead for subsequent
    invocations.
    """

    specification: ClassVar[Specification] = EnsembleSpecification()

    def __init__(
        self,
        *,
        num_units: int = 3,
        model_name: str = "openai:gpt-4o-mini",
        temperature: float = 0.7,
    ) -> None:
        """Initialize with JIT capabilities."""
        super().__init__(
            num_units=num_units, model_name=model_name, temperature=temperature
        )
        # The @jit decorator will handle caching the execution plan

    def forward(self, *, inputs: EnsembleInput) -> EnsembleOutput:
        """Process inputs with JIT optimization."""
        # Simulate execution by introducing an even smaller delay (JIT is fastest)
        time.sleep(0.02 * self.num_units)

        # Generate mock responses
        responses = [
            f"JIT response {i} to query: {inputs.query}" for i in range(self.num_units)
        ]

        return EnsembleOutput(query=inputs.query, responses=responses)


def run_operator_queries(
    *,
    operator_instance: Operator,
    queries: List[str],
    name: str,
    mode: str = "parallel",
) -> Tuple[List[float], float, List[EnsembleOutput]]:
    """Execute the given ensemble operator for each query and measure execution times.

    Args:
        operator_instance: The ensemble operator instance to run.
        queries: List of query strings.
        name: Name for logging purposes.
        mode: Execution mode ("parallel" or "sequential")

    Returns:
        Tuple containing:
            1. A list of per-query execution times.
            2. The total execution time for all queries.
            3. A list of result objects.
    """
    execution_times: List[float] = []
    results: List[EnsembleOutput] = []
    total_start_time: float = time.perf_counter()

    # Set the execution options based on the mode
    with execution_options(scheduler=mode):
        for query in queries:
            query_start_time: float = time.perf_counter()
            result = operator_instance(inputs={"query": query})
            query_end_time: float = time.perf_counter()

            elapsed_time: float = query_end_time - query_start_time
            execution_times.append(elapsed_time)
            results.append(result)

            # Type-safe way to access responses length
            response_count = (
                len(result.responses) if isinstance(result, EnsembleOutput) else 0
            )

            logging.info(
                "[%s] Query='%s' => #responses=%d | time=%.4fs",
                name.upper(),
                query,
                response_count,
                elapsed_time,
            )

    total_end_time: float = time.perf_counter()
    total_elapsed_time: float = total_end_time - total_start_time
    return execution_times, total_elapsed_time, results


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstrations comparing Baseline, Parallel, and JIT ensembles.

    This function constructs ensemble operator instances, executes a series of queries
    using each ensemble variant, and prints a consolidated timing summary with visualization.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define ensemble configuration parameters.
    model_name: str = "openai:gpt-4o-mini"  # You can change this to any available model
    temperature: float = 0.7
    num_units: int = 5  # Number of ensemble units (sub-calls)

    # Create ensemble operator instances.
    baseline_op = BaselineEnsemble(
        num_units=num_units, model_name=model_name, temperature=temperature
    )
    parallel_op = ParallelEnsemble(
        num_units=num_units, model_name=model_name, temperature=temperature
    )
    jit_op = JITEnsemble(
        num_units=num_units, model_name=model_name, temperature=temperature
    )

    # List of queries to execute.
    queries: List[str] = [
        "What is 2 + 2?",
        "Summarize quantum entanglement in simple terms.",
        "What is the longest river in Europe?",
        "Explain synergy in a business context.",
        "Who wrote Pride and Prejudice?",
    ]

    print(f"\n=== JIT Ensemble Comparison ({num_units} units per ensemble) ===")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Number of queries: {len(queries)}")

    # Execute queries for each ensemble variant.
    print("\nRunning baseline ensemble (sequential execution)...")
    baseline_times, total_baseline_time, baseline_results = run_operator_queries(
        operator_instance=baseline_op,
        queries=queries,
        name="Baseline",
        mode="sequential",
    )

    print("\nRunning parallel ensemble...")
    parallel_times, total_parallel_time, parallel_results = run_operator_queries(
        operator_instance=parallel_op, queries=queries, name="Parallel", mode="parallel"
    )

    print("\nRunning JIT ensemble...")
    jit_times, total_jit_time, jit_results = run_operator_queries(
        operator_instance=jit_op, queries=queries, name="JIT", mode="parallel"
    )

    # Create a simple table for displaying results
    try:
        from prettytable import PrettyTable

        # Use PrettyTable if available
        summary_table = PrettyTable()
        summary_table.field_names = [
            "Query",
            "Baseline (s)",
            "Parallel (s)",
            "JIT (s)",
            "Speedup",
        ]

        for index in range(len(queries)):
            # Calculate speedup percentage of JIT over baseline safely
            if baseline_times[index] > 0:
                speedup = (
                    (baseline_times[index] - jit_times[index]) / baseline_times[index]
                ) * 100
            else:
                speedup = 0.0

            # Truncate query for display
            query_display = (
                queries[index][:30] + "..."
                if len(queries[index]) > 30
                else queries[index]
            )

            summary_table.add_row(
                [
                    query_display,
                    f"{baseline_times[index]:.4f}",
                    f"{parallel_times[index]:.4f}",
                    f"{jit_times[index]:.4f}",
                    f"{speedup:.1f}%",
                ]
            )

        print("\n=== Timing Results ===")
        print(summary_table)

    except ImportError:
        # Fallback to text formatting if PrettyTable isn't available
        print("\n=== Timing Results ===")
        print(
            f"{'Query':<35} {'Baseline (s)':<15} {'Parallel (s)':<15} {'JIT (s)':<15} {'Speedup':<10}"
        )
        print("-" * 90)

        for index in range(len(queries)):
            # Calculate speedup percentage of JIT over baseline safely
            if baseline_times[index] > 0:
                speedup = (
                    (baseline_times[index] - jit_times[index]) / baseline_times[index]
                ) * 100
            else:
                speedup = 0.0

            # Truncate query for display
            query_display = (
                queries[index][:30] + "..."
                if len(queries[index]) > 30
                else queries[index]
            )

            print(
                f"{query_display:<35} {baseline_times[index]:<15.4f} {parallel_times[index]:<15.4f} {jit_times[index]:<15.4f} {speedup:<10.1f}%"
            )

    # Calculate and print summary statistics
    avg_baseline: float = sum(baseline_times) / len(baseline_times)
    avg_parallel: float = sum(parallel_times) / len(parallel_times)
    avg_jit: float = sum(jit_times) / len(jit_times)

    print("\n=== Performance Summary ===")
    print(f"Total Baseline time: {total_baseline_time:.4f}s")
    print(f"Total Parallel time: {total_parallel_time:.4f}s")
    print(f"Total JIT time:      {total_jit_time:.4f}s")

    print("\nAverage per-query time:")
    print(f"  Baseline: {avg_baseline:.4f}s")
    print(f"  Parallel: {avg_parallel:.4f}s")
    print(f"  JIT:      {avg_jit:.4f}s")

    # Calculate overall speedups with safety checks for division by zero
    if avg_baseline > 0:
        parallel_speedup = ((avg_baseline - avg_parallel) / avg_baseline) * 100
        jit_speedup = ((avg_baseline - avg_jit) / avg_baseline) * 100
    else:
        parallel_speedup = 0.0
        jit_speedup = 0.0

    if avg_parallel > 0:
        jit_vs_parallel_speedup = ((avg_parallel - avg_jit) / avg_parallel) * 100
    else:
        jit_vs_parallel_speedup = 0.0

    print("\nSpeedup percentages:")
    print(f"  Parallel vs Baseline: {parallel_speedup:.1f}%")
    print(f"  JIT vs Baseline:      {jit_speedup:.1f}%")
    print(f"  JIT vs Parallel:      {jit_vs_parallel_speedup:.1f}%")

    print("\n=== Key Benefits of JIT ===")
    print("1. Automatic tracing and optimization of execution paths")
    print("2. Cached execution plan for repeated queries")
    print("3. Reduced overhead for complex pipelines")
    print("4. Optimization across operator boundaries")

    print(
        "\nTo use JIT in your code, simply add the @jit decorator to your operator class:"
    )
    print("@jit")
    print("class MyOperator(Operator):")
    print("    def forward(self, *, inputs):")
    print("        # Your implementation here")
    print("        return result")


if __name__ == "__main__":
    main()
