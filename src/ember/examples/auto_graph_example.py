"""Automatic Graph Building Example.

This example demonstrates the enhanced JIT API with automatic graph building.
It shows how applying @jit to operators enables automatic graph building and
parallel execution without requiring manual graph construction.
"""

import logging
import time
from typing import Any, Dict, List, Optional

# ember imports
from ember.core.non import UniformEnsemble as Ensemble, MostCommon
from ember.core.registry.operator.base.operator_base import Operator
from ember.xcs.tracer.tracer_decorator import jit
from ember.xcs.engine.execution_options import execution_options


###############################################################################
# JIT-Decorated Operators
###############################################################################


@jit(sample_input={"query": "What is the capital of France?"})
class AutoEnsemble(Ensemble):
    """Ensemble with automatic graph building.

    The @jit decorator enables automatic graph building and parallel execution
    without requiring manual graph construction.
    """

    pass


@jit()
class AutoMostCommon(MostCommon):
    """Most common with automatic graph building."""

    pass


###############################################################################
# Pipeline Class
###############################################################################
@jit()
class AutoGraphPipeline(Operator):
    """Pipeline that demonstrates automatic graph building.

    This pipeline internally uses an ensemble and aggregator, but doesn't
    require manual graph construction. The @jit decorator handles this
    automatically, building a graph based on the actual execution trace.
    """

    specification = None  # Define a minimal specification to satisfy Operator requirements

    def __init__(
        self,
        *,
        model_name: str = "openai:gpt-4o-mini",
        num_units: int = 3,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the pipeline with configurable parameters.

        Args:
            model_name: The model to use
            num_units: Number of ensemble units
            temperature: Temperature for generation
        """
        self.ensemble = AutoEnsemble(
            num_units=num_units, model_name=model_name, temperature=temperature
        )
        self.aggregator = AutoMostCommon()

        # Create minimal specification for validation
        from ember.core.registry.prompt_specification.specification import Specification

        self.specification = Specification(input_model=None, output_model=None)

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline on the given inputs.

        Args:
            inputs: Dictionary containing the query

        Returns:
            Dictionary with aggregated results
        """
        # Pass the query to the ensemble
        ensemble_result = self.ensemble(inputs=inputs)

        # Combine the original query with ensemble responses for aggregation
        aggregator_inputs = {
            "query": inputs["query"],
            "responses": ensemble_result["responses"],
        }

        # Aggregate and return the final result
        return self.aggregator(inputs=aggregator_inputs)


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of automatic graph building."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Automatic Graph Building Example ===\n")

    # Create the pipeline
    pipeline = AutoGraphPipeline(model_name="openai:gpt-4o-mini", num_units=5)

    # Example queries to demonstrate caching and reuse
    queries = [
        "What is the capital of France?",
        "What is the tallest mountain in the world?",
        "Who wrote Romeo and Juliet?",
    ]

    print("First run - expect graph building overhead:")
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")

        start_time = time.perf_counter()
        result = pipeline(inputs={"query": query})
        elapsed = time.perf_counter() - start_time

        print(f"Answer: {result['final_answer']}")
        print(f"Time: {elapsed:.4f}s")

    print("\nRepeat first query to demonstrate cached execution:")
    start_time = time.perf_counter()
    result = pipeline(inputs={"query": queries[0]})
    elapsed = time.perf_counter() - start_time

    print(f"Answer: {result['final_answer']}")
    print(f"Time: {elapsed:.4f}s")

    print("\nUsing execution_options to control parallel execution:")
    with execution_options(scheduler="sequential"):
        start_time = time.perf_counter()
        result = pipeline(inputs={"query": "What is the speed of light?"})
        elapsed = time.perf_counter() - start_time

        print(f"Answer: {result['final_answer']}")
        print(f"Time: {elapsed:.4f}s (sequential execution)")


if __name__ == "__main__":
    main()
