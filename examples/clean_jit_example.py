"""Clean JIT API Example.

This module demonstrates the current JIT API for Ember, focusing on:
1. JIT decoration of individual operators with @jit for tracing
2. Manual graph construction for execution
3. Using TopologicalSchedulerWithParallelDispatch for optimized parallel execution

Note: Currently, each operator needs to be JIT-decorated separately, and graphs
must be built manually. Future versions will simplify this process.
"""

import logging
import time
from typing import Any, Dict, List

# ember imports
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.non import Ensemble, MostCommon
from ember.xcs.tracer.tracer_decorator import jit
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine.xcs_engine import (
    execute_graph,
    TopologicalSchedulerWithParallelDispatch,
)


###############################################################################
# JIT-Decorated Operators
###############################################################################


@jit()
class JITEnsemble(Ensemble):
    """Ensemble with JIT tracing.

    The @jit decorator enables tracing of this operator's execution,
    which can be used to optimize parallel execution.
    """

    pass


@jit()
class JITMostCommon(MostCommon):
    """Most common operator with JIT tracing."""

    pass


###############################################################################
# Simple Pipeline
###############################################################################
def run_pipeline(query: str, num_units: int = 3) -> Dict[str, Any]:
    """Run a simple pipeline with JIT-enabled operators.

    Args:
        query: The query to process
        num_units: Number of ensemble units

    Returns:
        The pipeline result
    """
    # Create the operators
    ensemble = JITEnsemble(
        num_units=num_units, model_name="openai:gpt-4o-mini", temperature=0.7
    )
    aggregator = JITMostCommon()

    # Create a graph for execution
    graph = XCSGraph()
    ensemble_id = graph.add_node(operator=ensemble, node_id="ensemble")
    aggregator_id = graph.add_node(operator=aggregator, node_id="aggregator")
    graph.add_edge(from_id=ensemble_id, to_id=aggregator_id)

    # Create optimized parallel scheduler
    scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=num_units)

    # Execute the graph with automatic parallelization
    start_time = time.perf_counter()
    result = execute_graph(
        graph=graph, global_input={"query": query}, scheduler=scheduler
    )
    end_time = time.perf_counter()

    logging.info(f"Pipeline execution took {end_time - start_time:.4f}s")

    return result


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of clean JIT API."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Clean JIT API Demonstration ===\n")

    # Example query
    query = "What is the capital of France?"

    print(f"Processing query: {query}")
    result = run_pipeline(query=query, num_units=5)

    print(f"\nFinal answer: {result['final_answer']}")
    print(f"Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    main()
