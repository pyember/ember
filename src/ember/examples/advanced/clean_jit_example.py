"""Clean JIT API Example.

This module demonstrates the current JIT API for Ember, focusing on:
1. JIT decoration of individual operators with @jit for tracing
2. Manual graph construction for execution
3. Using TopologicalSchedulerWithParallelDispatch for optimized parallel execution

Note: Currently, each operator needs to be JIT-decorated separately, and graphs
must be built manually. Future versions will simplify this process.

To run:
    uv run python src/ember/examples/advanced/clean_jit_example.py
"""

import logging
import time
from typing import ClassVar, List, Type

from ember.api import models
from ember.api.xcs import jit

# ember imports
from ember.core.registry.operator.base.operator_base import Operator, Specification
from ember.core.types.ember_model import EmberModel, Field

###############################################################################
# Input/Output Models
###############################################################################


class GenerationInput(EmberModel):
    """Input model for LLM operator.

    Attributes:
        query: The query to be processed.
    """

    query: str = Field(description="The query to be processed")


class GenerationOutput(EmberModel):
    """Output model for LLM operator.

    Attributes:
        responses: List of responses from the language model.
    """

    responses: List[str] = Field(
        description="List of responses from the language model"
    )


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


class GenerationSpecification(Specification):
    """Specification for LLM operator."""

    input_model: Type[EmberModel] = GenerationInput
    structured_output: Type[EmberModel] = GenerationOutput


class AggregatorSpecification(Specification):
    """Specification for aggregator operator."""

    input_model: Type[EmberModel] = AggregatorInput
    structured_output: Type[EmberModel] = AggregatorOutput


###############################################################################
# JIT-Decorated Operators
###############################################################################


@jit
class LLMOperator(Operator):
    """A language model operator that generates responses to a query."""

    specification: ClassVar[Specification] = GenerationSpecification()
    model_id: str

    def __init__(self, *, model_id: str = "anthropic:claude-3-5-sonnet") -> None:
        """Initialize the LLM operator.

        Args:
            model_id: The model identifier to use for generation.
        """
        self.model_id = model_id
        # Initialize the registry to get model
        self.registry = models.get_registry()

    def forward(self, *, inputs: GenerationInput) -> GenerationOutput:
        """Generate responses using the language model.

        Args:
            inputs: The input containing the query.

        Returns:
            Generated responses from the language model.
        """
        logging.info(f"Processing query with {self.model_id}: {inputs.query}")

        try:
            # Check if the model is available
            if self.registry.is_registered(self.model_id):
                # In a real application, we would call the model
                # model = self.registry.get_model(self.model_id)
                # response = model(prompt=inputs.query)

                # For demo purposes, we'll simulate a response to avoid API costs
                logging.info(f"Model {self.model_id} found, simulating response")
                time.sleep(0.1)  # Simulate API call latency
                simulated_responses = [
                    f"Answer about '{inputs.query}' from {self.model_id}",
                    f"Alternative perspective on '{inputs.query}'",
                    f"Additional context for '{inputs.query}'",
                ]
                return GenerationOutput(responses=simulated_responses)
            else:
                logging.warning(f"Model {self.model_id} not found in registry")
                return GenerationOutput(
                    responses=[f"Error: Model {self.model_id} not available"]
                )
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return GenerationOutput(responses=[f"Error: {str(e)}"])


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
        logging.info(f"Aggregating {len(inputs.responses)} responses")
        time.sleep(0.05)  # Simulate processing

        # In a real application, this would implement a more sophisticated
        # aggregation strategy, potentially using another LLM call
        if not inputs.responses:
            return AggregatorOutput(
                final_answer="No responses to aggregate", confidence=0.0
            )

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

    Raises:
        RuntimeError: If there is an error running the pipeline
    """
    # Create the operators
    llm_op = LLMOperator(model_id="anthropic:claude-3-5-sonnet")
    aggregator = AggregatorOperator()

    try:
        # Step 1: Run the LLM operator
        llm_response = llm_op(inputs={"query": query})
        logging.info(f"LLM operator output: {llm_response}")

        # Step 2: Feed the responses to the aggregator
        final_result = aggregator(inputs={"responses": llm_response.responses})
        logging.info(f"Aggregator output: {final_result}")

        return final_result
    except Exception as e:
        logging.error(f"Error in run_pipeline: {e}")
        # Create a fallback response in case of error
        error_result = AggregatorOutput(
            final_answer=f"Error processing query: {str(e)}", confidence=0.0
        )
        return error_result


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of clean JIT API."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("=== Clean JIT API Demonstration ===")

    # Example query
    query = "What is the capital of France?"

    logging.info(f"Processing query: {query}")
    result = run_pipeline(query=query, num_units=5)

    logging.info(f"Final answer: {result.final_answer}")
    logging.info(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    main()
