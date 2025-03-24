"""Automatic Graph Building Example.

This example demonstrates the enhanced JIT API with automatic graph building.
It shows how applying @jit to operators enables automatic graph building and
parallel execution without requiring manual graph construction.

To run:
    uv run python -m src.ember.examples.xcs.auto_graph_example
"""

import logging
import time
from typing import ClassVar, Optional, Type

# simplified import from ember API
from ember.api.xcs import execution_options, jit

# Import pre-built operators from standard library
from ember.core import non

# ember imports
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel

###############################################################################
# Input/Output Models
###############################################################################


class QueryInput(EmberModel):
    """Input model for query operations.

    Attributes:
        query: The text query to be processed.
    """

    query: str


class AggregationOutput(EmberModel):
    """Output model for aggregation operations.

    Attributes:
        final_answer: The aggregated response.
    """

    final_answer: Optional[str]


# Specification for pipeline
class PipelineSpecification(Specification):
    """Specification for the full pipeline."""

    input_model: Type[EmberModel] = QueryInput
    structured_output: Type[EmberModel] = AggregationOutput


###############################################################################
# Pipeline Classes
###############################################################################


class SimplePipeline(Operator[QueryInput, AggregationOutput]):
    """Pipeline that demonstrates automatic graph building.

    This pipeline internally uses an ensemble and aggregator, but doesn't
    use JIT yet. It will be a baseline for comparison.
    """

    # Class-level specification
    specification: ClassVar[Specification] = PipelineSpecification()

    # Field declarations
    ensemble: non.UniformEnsemble
    aggregator: non.MostCommon

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
        self.ensemble = non.UniformEnsemble(
            num_units=num_units, model_name=model_name, temperature=temperature
        )
        self.aggregator = non.MostCommon()

    def forward(self, *, inputs: QueryInput) -> AggregationOutput:
        """Execute the pipeline on the given inputs.

        Args:
            inputs: Contains the query to process

        Returns:
            The aggregated result with final answer
        """
        # Create dictionary input for the ensemble
        ensemble_inputs = {"query": inputs.query}

        # Execute the ensemble operator
        ensemble_result = self.ensemble.forward(inputs=ensemble_inputs)

        # Create input for the aggregator
        aggregator_inputs = {
            "query": inputs.query,
            "responses": ensemble_result["responses"],
        }

        # Execute the aggregator operator
        aggregator_result = self.aggregator.forward(inputs=aggregator_inputs)

        # Return the final result
        return AggregationOutput(final_answer=aggregator_result["final_answer"])


###############################################################################
# JIT-Decorated Pipeline
###############################################################################


@jit
class JITPipeline(Operator[QueryInput, AggregationOutput]):
    """Pipeline with JIT optimization.

    This pipeline uses the @jit decorator to enable automatic graph building and
    optimization.
    """

    # Class-level specification
    specification: ClassVar[Specification] = PipelineSpecification()

    # Field declarations
    ensemble: non.UniformEnsemble
    aggregator: non.MostCommon

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
        # Create component operators
        self.ensemble = non.UniformEnsemble(
            num_units=num_units, model_name=model_name, temperature=temperature
        )
        self.aggregator = non.MostCommon()

    def forward(self, *, inputs: QueryInput) -> AggregationOutput:
        """Execute the pipeline on the given inputs with JIT optimization.

        Args:
            inputs: Contains the query to process

        Returns:
            The aggregated result with final answer
        """
        # Create dictionary input for the ensemble
        ensemble_inputs = {"query": inputs.query}

        # Execute the ensemble operator
        ensemble_result = self.ensemble.forward(inputs=ensemble_inputs)

        # Create input for the aggregator
        aggregator_inputs = {
            "query": inputs.query,
            "responses": ensemble_result["responses"],
        }

        # Execute the aggregator operator
        aggregator_result = self.aggregator.forward(inputs=aggregator_inputs)

        # Return the final result
        return AggregationOutput(final_answer=aggregator_result["final_answer"])


###############################################################################
# Main Demonstration
###############################################################################


def main() -> None:
    """Run demonstration of automatic graph building."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Automatic Graph Building Example ===\n")

    # Example queries to demonstrate caching and reuse
    queries = [
        "What is the capital of France?",
        "What is the tallest mountain in the world?",
        "Who wrote Romeo and Juliet?",
    ]

    # First, regular pipeline without JIT
    print("Running baseline (no JIT):")
    baseline_pipeline = SimplePipeline(model_name="openai:gpt-4o-mini", num_units=3)

    for i, query in enumerate(queries[:1]):  # Just run one query as baseline
        print(f"\nQuery {i+1}: {query}")

        # Create typed input
        query_input = QueryInput(query=query)

        start_time = time.perf_counter()
        result = baseline_pipeline(inputs=query_input)
        elapsed = time.perf_counter() - start_time

        print(f"Answer: {result.final_answer}")
        print(f"Time: {elapsed:.4f}s")

    baseline_time = elapsed

    # Now with JIT
    print("\nRunning with JIT optimization:")
    jit_pipeline = JITPipeline(model_name="openai:gpt-4o-mini", num_units=3)

    print("\nFirst run - expect graph building overhead:")

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")

        # Create typed input
        query_input = QueryInput(query=query)

        start_time = time.perf_counter()
        result = jit_pipeline(inputs=query_input)
        elapsed = time.perf_counter() - start_time

        print(f"Answer: {result.final_answer}")
        print(f"Time: {elapsed:.4f}s")

        # Save time of first query for comparison
        if i == 0:
            first_jit_time = elapsed

    # Repeat first query to demonstrate caching benefits
    print("\nRepeat first query to demonstrate cached execution:")
    query_input = QueryInput(query=queries[0])

    start_time = time.perf_counter()
    result = jit_pipeline(inputs=query_input)
    cached_time = time.perf_counter() - start_time

    print(f"Answer: {result.final_answer}")
    print(f"Time: {cached_time:.4f}s")

    # With execution options for sequential execution
    print("\nUsing execution_options to control execution:")
    with execution_options(use_parallel=False):
        query_input = QueryInput(query="What is the speed of light?")

        start_time = time.perf_counter()
        result = jit_pipeline(inputs=query_input)
        seq_time = time.perf_counter() - start_time

        print(f"Answer: {result.final_answer}")
        print(f"Time: {seq_time:.4f}s (sequential execution)")

    # Print summary
    print("\n=== Performance Summary ===")
    print(f"Baseline time: {baseline_time:.4f}s")
    print(f"First JIT run: {first_jit_time:.4f}s (includes compilation)")
    print(f"Cached JIT run: {cached_time:.4f}s")
    print(f"Sequential JIT: {seq_time:.4f}s")

    if cached_time > 0 and baseline_time > 0:
        speedup = (baseline_time - cached_time) / baseline_time * 100
        print(f"\nJIT Speedup: {speedup:.1f}% faster than baseline")

    if cached_time > 0 and first_jit_time > 0:
        cache_benefit = (first_jit_time - cached_time) / first_jit_time * 100
        print(f"Caching Benefit: {cache_benefit:.1f}% faster after first run")

    print("\n=== Key Benefits of JIT ===")
    print("1. Automatic operator dependency analysis")
    print("2. Optimized execution with caching")
    print("3. Reduced overhead for complex pipelines")
    print("4. No manual graph construction required")


if __name__ == "__main__":
    main()
