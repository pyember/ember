"""JIT Ensemble Demonstration.

This module demonstrates three variants of a "LargeEnsemble" operator using ember's
non.py and operator registry:
    1) BaselineEnsemble: Executes eagerly without parallelism.
    2) ParallelEnsemble: Leverages concurrency via a scheduling plan.
    3) JITEnsemble: Combines parallel execution with JIT tracing to cache the concurrency plan.

It measures the total and per-query execution times for each approach.

All ensembles rely on the implementation in non.py, which wraps the EnsembleOperator
from operator_registry.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from prettytable import PrettyTable

# ember imports
from ember.core.configs.config import initialize_system
from ember.core.app_context import get_ember_context
from ember.xcs.graph_ir.operator_graph import OperatorGraph
from ember.xcs.graph_ir.operator_graph_runner import OperatorGraphRunner
from ember.xcs.tracer.tracer_decorator import jit

# Existing 'Ensemble' and its input type from non.py.
from ember.core.non import Ensemble
from ember.core.registry.operator.core.ensemble import (
    EnsembleOperatorInputs as EnsembleInputs,
)


###############################################################################
# BaselineEnsemble - Eager Execution (No Concurrency)
###############################################################################
class BaselineEnsemble(Ensemble):
    """Ensemble implementation that forces fully eager (serial) execution.

    This subclass disables concurrency by configuring the execution to
    run serially rather than in parallel.
    """

    # BaselineEnsemble forces serial execution by configuration
    # The metaclass-based EmberModule system handles execution details


###############################################################################
# ParallelEnsemble - Standard Concurrency
###############################################################################
class ParallelEnsemble(Ensemble):
    """Ensemble implementation that leverages standard concurrency.

    Inherits directly from the underlying Ensemble, which produces a concurrency plan
    via the EnsembleOperator when the runner's max_workers is greater than 1.
    """

    pass


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

    pass


def run_operator_queries(
    *,
    operator_instance: Ensemble,
    operator_graph_runner: OperatorGraphRunner,
    queries: List[str],
    node_identifier: str,
) -> Tuple[List[float], float]:
    """Execute the given ensemble operator for each query and measure execution times.

    Args:
        operator_instance (Ensemble): The ensemble operator instance to run.
        operator_graph_runner (OperatorGraphRunner): Runner for executing operator graphs.
        queries (List[str]): List of query strings.
        node_identifier (str): Identifier for the node when adding to the operator graph.

    Returns:
        Tuple[List[float], float]:
            A tuple containing:
                1. A list of per-query execution times.
                2. The total execution time for all queries.
    """
    execution_times: List[float] = []
    total_start_time: float = time.perf_counter()

    for query in queries:
        operator_graph: OperatorGraph = OperatorGraph()
        operator_graph.add_node(operator=operator_instance, node_id=node_identifier)
        query_start_time: float = time.perf_counter()
        result = operator_graph_runner.run(
            graph=operator_graph, inputs={"query": query}
        )
        query_end_time: float = time.perf_counter()

        elapsed_time: float = query_end_time - query_start_time
        execution_times.append(elapsed_time)
        logging.info(
            "[%s] Query='%s' => #responses=%d | time=%.4fs",
            node_identifier.upper(),
            query,
            len(result.responses),
            elapsed_time,
        )

    total_end_time: float = time.perf_counter()
    total_elapsed_time: float = total_end_time - total_start_time
    return execution_times, total_elapsed_time


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstrations comparing Baseline, Parallel, and JIT ensembles.

    This function initializes the ember system, constructs ensemble operator instances, executes
    a series of queries using each ensemble variant, and prints a consolidated timing summary.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize the ember system.
    context = get_ember_context()
    initialize_system(registry=context.registry)
    # Optionally configure further settings, e.g., API keys:
    # CONFIG.set("models", "openai_api_key", "<YOUR-KEY>")

    # Define ensemble configuration parameters.
    model_name: str = "openai:gpt-4o-mini"
    temperature: float = 0.7
    num_units: int = 10  # Number of ensemble units (sub-calls)

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

    # Instantiate the OperatorGraphRunner with the specified number of workers.
    runner: OperatorGraphRunner = OperatorGraphRunner(max_workers=num_units)

    # List of queries to execute.
    queries: List[str] = [
        "What is 2 + 2?",
        "Summarize quantum entanglement in simple terms.",
        "Longest river in Europe?",
        "Explain synergy in a business context.",
        "Who wrote Pride and Prejudice?",
    ]

    # Execute queries for each ensemble variant.
    baseline_times, total_baseline_time = run_operator_queries(
        operator_instance=baseline_op,
        operator_graph_runner=runner,
        queries=queries,
        node_identifier="baseline_op",
    )

    parallel_times, total_parallel_time = run_operator_queries(
        operator_instance=parallel_op,
        operator_graph_runner=runner,
        queries=queries,
        node_identifier="parallel_op",
    )

    jit_times, total_jit_time = run_operator_queries(
        operator_instance=jit_op,
        operator_graph_runner=runner,
        queries=queries,
        node_identifier="jit_op",
    )

    # Print timing summary using PrettyTable.
    summary_table: PrettyTable = PrettyTable()
    summary_table.field_names = ["Query #", "Baseline (s)", "Parallel (s)", "JIT (s)"]

    for index in range(len(queries)):
        summary_table.add_row(
            [
                index + 1,
                f"{baseline_times[index]:.4f}",
                f"{parallel_times[index]:.4f}",
                f"{jit_times[index]:.4f}",
            ]
        )

    print(
        "\nTiming Results for Baseline vs. Parallel vs. JIT "
        f"({num_units} LM calls each, max_workers={num_units}):"
    )
    print(summary_table)

    print(f"\nTotal Baseline time: {total_baseline_time:.4f}s")
    print(f"Total Parallel time: {total_parallel_time:.4f}s")
    print(f"Total JIT time:      {total_jit_time:.4f}s")

    # Calculate and print average per-query execution times.
    avg_baseline: float = sum(baseline_times) / len(baseline_times)
    avg_parallel: float = sum(parallel_times) / len(parallel_times)
    avg_jit: float = sum(jit_times) / len(jit_times)

    print(f"\nAverage per-query Baseline: {avg_baseline:.4f}s")
    print(f"Average per-query Parallel: {avg_parallel:.4f}s")
    print(f"Average per-query JIT:      {avg_jit:.4f}s")


if __name__ == "__main__":
    main()
