"""Example architectures demonstrating clean Ember operator composition patterns.

This module showcases best practices for defining and composing operators
in Ember.
"""

import logging
from typing import ClassVar, Type

# Import the non module directly from ember core
from ember.core import non
from ember.core.app_context import get_ember_context
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field

# Configure logging
logger = logging.getLogger(__name__)


class NetworkInput(EmberModel):
    """Input model for network operators.

    Attributes:
        query: The query to process
    """

    query: str = Field(description="The query to process")


class NetworkOutput(EmberModel):
    """Output model for network operators.

    Attributes:
        final_answer: The final processed answer
    """

    final_answer: str = Field(description="The final processed answer")


class SubNetworkSpecification(Specification):
    """Specification for SubNetwork operator."""

    # Use the pattern established in NestedNetworkSpecification
    input_model: Type[EmberModel] = NetworkInput
    structured_output: Type[EmberModel] = NetworkOutput


class SubNetwork(Operator[NetworkInput, NetworkOutput]):
    """SubNetwork that composes an ensemble with verification.

    This operator first processes inputs through an ensemble of models and subsequently verifies
    the output based on the initial ensemble's response.

    Attributes:
        specification: The operator's input/output contract
        ensemble: A uniform ensemble with N units of the specified model
        verifier: A verification operator using the specified model
    """

    specification: ClassVar[Specification] = SubNetworkSpecification()
    ensemble: non.UniformEnsemble
    verifier: non.Verifier

    def __init__(
        self, *, model_name: str = "openai:gpt-4o", num_units: int = 2
    ) -> None:
        """Initialize the SubNetwork with configurable components.

        Args:
            model_name: The model to use for both ensemble and verification
            num_units: Number of ensemble units to run in parallel
        """
        logger.debug(
            f"Initializing SubNetwork with model={model_name}, units={num_units}"
        )
        self.ensemble = non.UniformEnsemble(
            num_units=num_units, model_name=model_name, temperature=0.0
        )
        self.verifier = non.Verifier(model_name=model_name, temperature=0.0)
        logger.debug("SubNetwork initialization complete")

    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        """Process the input through the ensemble and verify the results.

        Args:
            inputs: The validated input containing the query

        Returns:
            A NetworkOutput with the verified answer
        """
        logger.debug("Processing input through SubNetwork ensemble")
        ensemble_result = self.ensemble(query=inputs.query)

        # Extract the first response for verification
        candidate_answer = ensemble_result["responses"][0]
        logger.debug("Selected candidate answer from ensemble")

        logger.debug("Verifying candidate answer")
        verified_result = self.verifier(
            query=inputs.query, candidate_answer=candidate_answer
        )

        # Return structured output
        logger.debug("SubNetwork processing complete")
        return NetworkOutput(final_answer=verified_result["revised_answer"])


class NestedNetworkSpecification(Specification):
    """Specification for NestedNetwork operator."""

    input_model: Type[EmberModel] = NetworkInput
    structured_output: Type[EmberModel] = NetworkOutput


class NestedNetwork(Operator[NetworkInput, NetworkOutput]):
    """Nested network that aggregates results from multiple sub-networks and applies judgment.

    This operator executes two subnetwork branches and uses a judge operator to synthesize the outputs.

    Attributes:
        specification: The operator's input/output contract
        sub1: The first sub-network instance
        sub2: The second sub-network instance
        judge: A judge synthesis operator
    """

    specification: ClassVar[Specification] = NestedNetworkSpecification()
    sub1: SubNetwork
    sub2: SubNetwork
    judge: non.JudgeSynthesis

    def __init__(self, *, model_name: str = "openai:gpt-4o") -> None:
        """Initialize the NestedNetwork with sub-networks and a judge.

        Args:
            model_name: The model to use for all components
        """
        logger.debug(f"Initializing NestedNetwork with model={model_name}")
        self.sub1 = SubNetwork(model_name=model_name)
        self.sub2 = SubNetwork(model_name=model_name)
        self.judge = non.JudgeSynthesis(model_name=model_name, temperature=0.0)
        logger.debug("NestedNetwork initialization complete")

    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        """Execute the nested network by processing through sub-networks and judging results.

        Args:
            inputs: The validated input containing the query

        Returns:
            A NetworkOutput with the final judged answer
        """
        logger.debug("Starting NestedNetwork execution")

        # Process through parallel sub-networks
        logger.debug("Processing through first sub-network")
        s1_out = self.sub1(inputs=inputs)

        logger.debug("Processing through second sub-network")
        s2_out = self.sub2(inputs=inputs)

        logger.debug("Applying judge synthesis to sub-network outputs")
        judged_result = self.judge(
            query=inputs.query, responses=[s1_out.final_answer, s2_out.final_answer]
        )

        # Return structured output
        logger.debug("NestedNetwork execution complete")
        return NetworkOutput(final_answer=judged_result["synthesized_response"])


def create_nested_network(*, model_name: str = "gpt-4o") -> NestedNetwork:
    """Create a nested network with the specified model.

    Args:
        model_name: The model to use throughout the network

    Returns:
        A configured NestedNetwork operator
    """
    logger.info(f"Creating nested network with model: {model_name}")
    return NestedNetwork(model_name=model_name)


def create_pipeline(*, model_name: str = "gpt-4o") -> non.Sequential:
    """Create a declarative pipeline using the Sequential NON operator.

    This demonstrates a more declarative approach to building operator pipelines
    using the Sequential operator, which chains operators together automatically.

    Args:
        model_name: The model to use throughout the pipeline

    Returns:
        A callable pipeline accepting standardized inputs
    """
    logger.info(f"Creating declarative pipeline with model: {model_name}")

    # Create a pipeline using Sequential operator for cleaner composition
    pipeline = non.Sequential(
        operators=[
            # Generate 3 responses with the same model
            non.UniformEnsemble(
                num_units=3,
                model_name=model_name,
                temperature=0.7,  # Using higher temperature for diversity
            ),
            # Pass the ensemble responses to a judge for synthesis
            non.JudgeSynthesis(model_name=model_name, temperature=0.0),
            # Verify the synthesized response
            non.Verifier(model_name=model_name, temperature=0.0),
        ]
    )

    logger.debug("Pipeline created successfully")
    return pipeline


if __name__ == "__main__":
    # Use the centralized logging configuration with reduced verbosity
    from ember.core.utils.logging import configure_logging
    from ember.xcs.engine.execution_options import set_execution_options

    configure_logging(verbose=False)
    
    # Initialize the ember context
    context = get_ember_context()
    logger.info("Ember context initialized")

    # Example 1: Using the object-oriented approach
    logger.info("=== Object-Oriented Style ===")
    network = NestedNetwork(model_name="openai:gpt-4o")
    test_input = NetworkInput(
        query="What are three key principles of functional programming?"
    )
    logger.info(f"Running network with query: {test_input.query}")
    test_result = network(inputs=test_input)
    logger.info(f"Answer: {test_result.final_answer}")

    # Example 2: Using the declarative non.Sequential "pipeline" style
    logger.info("=== Declarative Pipeline Style ===")
    pipeline = create_pipeline(model_name="openai:gpt-4o")
    query = "What are three key principles of functional programming?"

    # For consistency, use kwargs pattern for pipeline invocation too
    logger.info(f"Running pipeline with query: {query}")
    result = pipeline(query=query)
    logger.info(f"Answer: {result['revised_answer']}")
