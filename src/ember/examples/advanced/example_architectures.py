"""Example architectures demonstrating clean Ember operator composition patterns.

This module showcases best practices for defining and composing operators
in Ember, following patterns from high-quality OSS libraries like JAX and PyTorch.
It uses the Network of Operators (NON) pattern for standardized, composable LLM pipelines.
"""

from typing import ClassVar, Dict, List, Type, Any

# Import the non module directly from ember
from ember import non
from ember.api.operators import Operator, EmberModel, Field
from ember.core.registry.specification.specification import Specification
from ember.core.app_context import get_ember_context


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

    def __init__(self, *, model_name: str = "gpt-4o", num_units: int = 2) -> None:
        """Initialize the SubNetwork with configurable components.

        Args:
            model_name: The model to use for both ensemble and verification
            num_units: Number of ensemble units to run in parallel
        """
        self.ensemble = non.UniformEnsemble(
            num_units=num_units, model_name=model_name, temperature=0.0
        )
        self.verifier = non.Verifier(model_name=model_name, temperature=0.0)

    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        """Process the input through the ensemble and verify the results.

        Args:
            inputs: The validated input containing the query

        Returns:
            A NetworkOutput with the verified answer
        """
        ensemble_result = self.ensemble(query=inputs.query)

        # Extract the first response for verification
        candidate_answer = ensemble_result["responses"][0]

        verified_result = self.verifier(
            query=inputs.query, candidate_answer=candidate_answer
        )

        # Return structured output
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

    def __init__(self, *, model_name: str = "gpt-4o") -> None:
        """Initialize the NestedNetwork with sub-networks and a judge.

        Args:
            model_name: The model to use for all components
        """
        self.sub1 = SubNetwork(model_name=model_name)
        self.sub2 = SubNetwork(model_name=model_name)
        self.judge = non.JudgeSynthesis(model_name=model_name, temperature=0.0)

    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        """Execute the nested network by processing through sub-networks and judging results.

        Args:
            inputs: The validated input containing the query

        Returns:
            A NetworkOutput with the final judged answer
        """
        # Process through parallel sub-networks
        s1_out = self.sub1(inputs=inputs)
        s2_out = self.sub2(inputs=inputs)

        judged_result = self.judge(
            query=inputs.query, responses=[s1_out.final_answer, s2_out.final_answer]
        )

        # Return structured output
        return NetworkOutput(final_answer=judged_result["synthesized_response"])


def create_nested_network(*, model_name: str = "gpt-4o") -> NestedNetwork:
    """Create a nested network with the specified model.

    Args:
        model_name: The model to use throughout the network

    Returns:
        A configured NestedNetwork operator
    """
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
    # Create a pipeline using Sequential operator for cleaner composition
    return non.Sequential(
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


if __name__ == "__main__":
    # Initialize the ember context
    context = get_ember_context()

    # Example 1: Using the object-oriented approach
    print("\n=== Object-Oriented Style ===")
    network = NestedNetwork(model_name="gpt-4o")
    test_input = NetworkInput(
        query="What are three key principles of functional programming?"
    )
    test_result = network(inputs=test_input)
    print(f"Query: {test_input.query}")
    print(f"Answer: {test_result.final_answer}\n")

    # Example 2: Using the declarative pipeline style
    print("=== Declarative Pipeline Style ===")
    pipeline = create_pipeline(model_name="gpt-4o")

    # For consistency, use kwargs pattern for pipeline invocation too
    result = pipeline(query="What are three key principles of functional programming?")
    print(f"Query: What are three key principles of functional programming?")
    print(f"Answer: {result['revised_answer']}\n")
