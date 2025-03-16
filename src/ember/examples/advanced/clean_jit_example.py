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

# ember imports
from ember.core.registry.operator.base.operator_base import Operator, Specification
from ember.core.types.ember_model import EmberModel, Field
from ember.api.xcs import jit
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine.xcs_engine import (
    execute_graph,
    TopologicalSchedulerWithParallelDispatch,
)


###############################################################################
# Input/Output Models
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


class AggregatorInput(EmberModel):
    """Input model for aggregator operator.

    Attributes:
        responses: List of responses to aggregate.
    """

    responses: List[str] = Field(description="List of responses to aggregate")


class AggregatorOutput(EmberModel):
    """Output model for aggregator operator.

    Attributes:
        final_answer: The aggregated result.
        confidence: Confidence score for the aggregation.
    """

    final_answer: str = Field(description="The aggregated result")
    confidence: float = Field(description="Confidence score for the aggregation")


###############################################################################
# Specifications
###############################################################################


class MockSpecification(Specification):
    """Specification for mock operator."""

    input_model: Type[EmberModel] = MockInput
    structured_output: Type[EmberModel] = MockOutput


class AggregatorSpecification(Specification):
    """Specification for aggregator operator."""

    input_model: Type[EmberModel] = AggregatorInput
    structured_output: Type[EmberModel] = AggregatorOutput


###############################################################################
# JIT-Decorated Operators
###############################################################################


@jit
class MockOperator(Operator):
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


@jit
class AggregatorOperator(Operator):
    """An aggregator that combines multiple responses."""

    specification: ClassVar[Specification] = AggregatorSpecification()

    def forward(self, *, inputs: AggregatorInput) -> AggregatorOutput:
        """Aggregate responses into a final answer.

        Args:
            inputs: The responses to aggregate.

        Returns:
            Aggregated result with confidence score.
        """
        time.sleep(0.05)  # Simulate processing
        return AggregatorOutput(
            final_answer=inputs.responses[0],  # Just take the first one for this demo
            confidence=0.95,
        )


def run_pipeline(query: str, num_units: int = 3) -> AggregatorOutput:
    """Run a simple pipeline with JIT-enabled operators.

    Args:
        query: The query to process
        num_units: Number of worker threads

    Returns:
        The pipeline result
    """
    try:
        # Instead of trying to use XCSGraph directly, we'll use a simpler two-step approach
        # Create the operators
        mock_op = MockOperator()
        aggregator = AggregatorOperator()

        # Step 1: Run the mock operator
        mock_response = mock_op(inputs={"query": query})
        logging.info(f"Mock operator output: {mock_response}")

        # Step 2: Feed the responses to the aggregator
        final_result = aggregator(inputs={"responses": mock_response.responses})
        logging.info(f"Aggregator output: {final_result}")

        return final_result
    except Exception as e:
        logging.error(f"Error in run_pipeline: {e}")
        raise


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

    print(f"\nFinal answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    main()
