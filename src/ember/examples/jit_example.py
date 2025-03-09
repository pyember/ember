"""JIT Ensemble Demonstration.

This module demonstrates three variants of a "LargeEnsemble" operator using ember's
API:
    1) BaselineEnsemble: Executes eagerly without parallelism.
    2) ParallelEnsemble: Leverages concurrency via a scheduling plan.
    3) JITEnsemble: Combines parallel execution with JIT tracing to cache the concurrency plan.

It measures the total and per-query execution times for each approach.

To run:
    poetry run python src/ember/examples/jit_example.py
"""

import logging
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# ember API imports
from ember.api import non
from ember.api.xcs import jit, execution_options


###############################################################################
# BaselineEnsemble - Eager Execution (No Concurrency)
###############################################################################
class BaselineEnsemble(non.Ensemble):
    """Ensemble implementation that forces fully eager (serial) execution.

    This subclass configures the execution to run serially rather than in parallel
    by using appropriate execution options.
    """
    
    def __init__(self, *, num_units: int = 3, model_name: str = "openai:gpt-4o-mini", temperature: float = 0.7) -> None:
        """Initialize with sequential execution options."""
        super().__init__(num_units=num_units, model_name=model_name, temperature=temperature)
        # Will be configured to run sequentially in the runner


###############################################################################
# ParallelEnsemble - Standard Concurrency
###############################################################################
class ParallelEnsemble(non.Ensemble):
    """Ensemble implementation that leverages standard concurrency.

    Inherits directly from the underlying Ensemble, which produces a concurrency plan
    when executed with parallel execution options.
    """

    def __init__(self, *, num_units: int = 3, model_name: str = "openai:gpt-4o-mini", temperature: float = 0.7) -> None:
        """Initialize with standard configuration."""
        super().__init__(num_units=num_units, model_name=model_name, temperature=temperature)
        # Will be configured to run in parallel in the runner


###############################################################################
# JITEnsemble - Parallel Execution with JIT Tracing
###############################################################################
@jit(sample_input={"query": "JIT Warmup Sample"})
class JITEnsemble(ParallelEnsemble):
    """Ensemble implementation with JIT tracing for optimized concurrency.

    Uses the same parallel approach as ParallelEnsemble but with JIT decoration. The first call
    (or __init__ if sample_input is provided) triggers tracing and caching of the concurrency plan,
    reducing overhead for subsequent invocations.
    """

    def __init__(self, *, num_units: int = 3, model_name: str = "openai:gpt-4o-mini", temperature: float = 0.7) -> None:
        """Initialize with JIT capabilities."""
        super().__init__(num_units=num_units, model_name=model_name, temperature=temperature)
        # The @jit decorator will handle caching the execution plan


def run_operator_queries(
    *,
    operator_instance: non.Ensemble,
    queries: List[str],
    name: str,
    mode: str = "parallel"
) -> Tuple[List[float], float, List[Dict[str, Any]]]:
    """Execute the given ensemble operator for each query and measure execution times.

    Args:
        operator_instance (non.Ensemble): The ensemble operator instance to run.
        queries (List[str]): List of query strings.
        name (str): Name for logging purposes.
        mode (str): Execution mode ("parallel" or "sequential")

    Returns:
        Tuple[List[float], float, List[Dict[str, Any]]]:
            A tuple containing:
                1. A list of per-query execution times.
                2. The total execution time for all queries.
                3. A list of result objects.
    """
    execution_times: List[float] = []
    results: List[Dict[str, Any]] = []
    total_start_time: float = time.perf_counter()

    # Set the execution options based on the mode
    ctx_manager = execution_options(scheduler=mode)
    
    with ctx_manager:
        for query in queries:
            query_start_time: float = time.perf_counter()
            result = operator_instance(inputs={"query": query})
            query_end_time: float = time.perf_counter()

            elapsed_time: float = query_end_time - query_start_time
            execution_times.append(elapsed_time)
            results.append(result)
            
            logging.info(
                "[%s] Query='%s' => #responses=%d | time=%.4fs",
                name.upper(),
                query,
                len(result["responses"]) if "responses" in result else 0,
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
    baseline_op: BaselineEnsemble = BaselineEnsemble(
        num_units=num_units, model_name=model_name, temperature=temperature
    )
    parallel_op: ParallelEnsemble = ParallelEnsemble(
        num_units=num_units, model_name=model_name, temperature=temperature
    )
    jit_op: JITEnsemble = JITEnsemble(
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
        mode="sequential"
    )

    print("\nRunning parallel ensemble...")
    parallel_times, total_parallel_time, parallel_results = run_operator_queries(
        operator_instance=parallel_op,
        queries=queries,
        name="Parallel",
        mode="parallel"
    )

    print("\nRunning JIT ensemble...")
    jit_times, total_jit_time, jit_results = run_operator_queries(
        operator_instance=jit_op,
        queries=queries,
        name="JIT",
        mode="parallel"
    )

    # Print timing summary using PrettyTable.
    summary_table: PrettyTable = PrettyTable()
    summary_table.field_names = ["Query", "Baseline (s)", "Parallel (s)", "JIT (s)", "Speedup"]

    for index in range(len(queries)):
        # Calculate speedup percentage of JIT over baseline
        speedup = ((baseline_times[index] - jit_times[index]) / baseline_times[index]) * 100
        
        # Truncate query for display
        query_display = queries[index][:30] + "..." if len(queries[index]) > 30 else queries[index]
        
        summary_table.add_row(
            [
                query_display,
                f"{baseline_times[index]:.4f}",
                f"{parallel_times[index]:.4f}",
                f"{jit_times[index]:.4f}",
                f"{speedup:.1f}%"
            ]
        )

    print("\n=== Timing Results ===")
    print(summary_table)

    # Calculate and print summary statistics
    avg_baseline: float = sum(baseline_times) / len(baseline_times)
    avg_parallel: float = sum(parallel_times) / len(parallel_times)
    avg_jit: float = sum(jit_times) / len(jit_times)
    
    print("\n=== Performance Summary ===")
    print(f"Total Baseline time: {total_baseline_time:.4f}s")
    print(f"Total Parallel time: {total_parallel_time:.4f}s")
    print(f"Total JIT time:      {total_jit_time:.4f}s")
    
    print(f"\nAverage per-query time:")
    print(f"  Baseline: {avg_baseline:.4f}s")
    print(f"  Parallel: {avg_parallel:.4f}s")
    print(f"  JIT:      {avg_jit:.4f}s")
    
    # Calculate overall speedups
    parallel_speedup = ((avg_baseline - avg_parallel) / avg_baseline) * 100
    jit_speedup = ((avg_baseline - avg_jit) / avg_baseline) * 100
    jit_vs_parallel_speedup = ((avg_parallel - avg_jit) / avg_parallel) * 100
    
    print(f"\nSpeedup percentages:")
    print(f"  Parallel vs Baseline: {parallel_speedup:.1f}%")
    print(f"  JIT vs Baseline:      {jit_speedup:.1f}%")
    print(f"  JIT vs Parallel:      {jit_vs_parallel_speedup:.1f}%")
    
    print("\n=== Key Benefits of JIT ===")
    print("1. Automatic tracing and optimization of execution paths")
    print("2. Cached execution plan for repeated queries")
    print("3. Reduced overhead for complex pipelines")
    print("4. Optimization across operator boundaries")
    
    print("\nTo use JIT in your code, simply add the @jit decorator to your operator class:")
    print("@jit()")
    print("class MyOperator(Operator):")
    print("    def forward(self, *, inputs):")
    print("        # Your implementation here")
    print("        return result")


if __name__ == "__main__":
    main()
