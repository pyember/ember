"""Clean JIT API Example.

This module demonstrates the current JIT API for Ember, focusing on:
1. JIT decoration of individual operators with @jit for tracing
2. Manual graph construction for execution
3. Using TopologicalSchedulerWithParallelDispatch for optimized parallel execution

Note: Currently, each operator needs to be JIT-decorated separately, and graphs
must be built manually. Future versions will simplify this process.

To run:
    poetry run python src/ember/examples/clean_jit_example.py
"""

import logging
import time
from typing import Any, ClassVar, Dict, List, Type

from pydantic import Field

# ember imports
from ember.api.operator import Operator, Specification
from ember.api.types import EmberModel
from ember.api import non
from ember.api.xcs import jit
from ember.api.xcs.graph import XCSGraph
from ember.api.xcs.engine import (
    execute_graph,
    TopologicalSchedulerWithParallelDispatch,
)


###############################################################################
# JIT-Decorated Operators
###############################################################################


@jit()
class JITEnsemble(non.UniformEnsemble):
    """Ensemble with JIT tracing.

    The @jit decorator enables tracing of this operator's execution,
    which can be used to optimize parallel execution.
    """
    pass


@jit()
class JITMostCommon(non.MostCommon):
    """Most common operator with JIT tracing."""
    pass


###############################################################################
# Simple Pipeline
###############################################################################
class MockInput(EmberModel):
    """Input model for mock operator.
    
    Attributes:
        query: The query to be processed.
    """
    query: str = Field(description="The query to be processed")

class MockOutput(EmberModel):
    """Output model for mock operator.
    
    Attributes:
        responses: List of responses from the mock operator.
    """
    responses: List[str] = Field(description="List of responses from the mock operator")

class MockSpecification(Specification):
    """Specification for mock operator."""
    input_model: Type[EmberModel] = MockInput
    output_model: Type[EmberModel] = MockOutput

@jit()
class MockOperator(Operator[MockInput, MockOutput]):
    """A mock operator for demonstration purposes."""
    
    specification: ClassVar[Specification] = MockSpecification()
    
    def forward(self, *, inputs: MockInput) -> MockOutput:
        """Mock response generation.
        
        Args:
            inputs: The input containing the query.
            
        Returns:
            Mock responses for demonstration.
        """
        time.sleep(0.1)  # Simulate API call
        return MockOutput(responses=["Answer A", "Answer B", "Answer C"])

def run_pipeline(query: str, num_units: int = 3) -> Dict[str, Any]:
    """Run a simple pipeline with JIT-enabled operators.

    Args:
        query: The query to process
        num_units: Number of ensemble units

    Returns:
        The pipeline result
    """
    # Create a mock operator instead of an actual LLM ensemble
    mock_op = MockOperator()
    
    # Create the aggregator
    aggregator = JITMostCommon()

    # Create a graph for execution
    graph = XCSGraph()
    mock_id = graph.add_node(operator=mock_op, node_id="mock_ensemble")
    aggregator_id = graph.add_node(operator=aggregator, node_id="aggregator")
    graph.add_edge(from_id=mock_id, to_id=aggregator_id)

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
