from typing import ClassVar, Dict, Optional, Union, cast, Type
from pydantic import Field

from ember.core.app_context import get_ember_context
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.types.ember_model import EmberModel
from ember.core import non
from ember.core.registry.prompt_specification.specification import Specification


class NetworkInput(EmberModel):
    """Input model for network operators."""

    query: str = Field(description="The query to process")


class NetworkOutput(EmberModel):
    """Output model for network operators."""

    final_answer: str = Field(description="The final processed answer")


class SubNetworkSpecification(Specification):
    """Specification for SubNetwork operator."""

    input_model: Type[EmberModel] = NetworkInput
    output_model: Type[EmberModel] = NetworkOutput


class SubNetwork(Operator[NetworkInput, NetworkOutput]):
    """SubNetwork that composes an ensemble with verification.

    This operator first processes inputs through an ensemble of models and subsequently verifies
    the output based on the initial ensemble's response.

    Attributes:
        ensemble (non.UniformEnsemble): An ensemble operator with 2 units of "gpt-4o".
        verifier (non.Verifier): A verification operator using "gpt-4o".
    """

    specification: ClassVar[Specification] = SubNetworkSpecification()
    ensemble: non.UniformEnsemble
    verifier: non.Verifier

    def __init__(self) -> None:
        """Initializes the SubNetwork with a specified ensemble and verification components."""
        super().__init__()
        self.ensemble = non.UniformEnsemble(
            num_units=2, model_name="gpt-4o", temperature=0.0
        )
        self.verifier = non.Verifier(model_name="gpt-4o", temperature=0.0)

    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        """Processes the input through the ensemble and applies verification.

        Args:
            inputs: The validated input containing the query.

        Returns:
            A NetworkOutput with the verified answer.
        """
        # Process through ensemble
        ensemble_input = {"query": inputs.query}
        ensemble_result = self.ensemble(inputs=ensemble_input)

        # Extract ensemble result explicitly
        if (
            not isinstance(ensemble_result, dict)
            or "final_answer" not in ensemble_result
        ):
            candidate_answer = ""
        else:
            candidate_answer = ensemble_result["final_answer"]

        # Prepare verification input
        verification_input = {
            "query": inputs.query,
            "candidate_answer": candidate_answer,
        }

        # Verify the ensemble's output
        verified_result = self.verifier(inputs=verification_input)

        # Extract result explicitly - no hidden error handling
        if (
            not isinstance(verified_result, dict)
            or "final_answer" not in verified_result
        ):
            final_answer = ""
        else:
            final_answer = verified_result["final_answer"]

        # Return structured output
        return NetworkOutput(final_answer=final_answer)


class NestedNetworkSpecification(Specification):
    """Specification for NestedNetwork operator."""

    input_model: Type[EmberModel] = NetworkInput
    output_model: Type[EmberModel] = NetworkOutput


class NestedNetwork(Operator[NetworkInput, NetworkOutput]):
    """Nested network that aggregates results from multiple sub-networks and applies a final judgment.

    This operator executes two subnetwork branches and uses a judge operator to synthesize the outputs.

    Attributes:
        sub1 (SubNetwork): The first sub-network instance.
        sub2 (SubNetwork): The second sub-network instance.
        judge (non.JudgeSynthesis): A judge synthesis operator using "gpt-4o".
    """

    specification: ClassVar[Specification] = NestedNetworkSpecification()
    sub1: SubNetwork
    sub2: SubNetwork
    judge: non.JudgeSynthesis

    def __init__(self) -> None:
        """Initializes the NestedNetwork with two SubNetwork instances and a final Judge operator."""
        super().__init__()
        self.sub1 = SubNetwork()
        self.sub2 = SubNetwork()
        self.judge = non.JudgeSynthesis(model_name="gpt-4o", temperature=0.0)

    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        """Executes the nested network by processing inputs through sub-networks and aggregating responses.

        Args:
            inputs: The validated input containing the query.

        Returns:
            A NetworkOutput with the final judged answer.
        """
        # Process through parallel sub-networks
        s1_out = self.sub1(inputs=inputs)
        s2_out = self.sub2(inputs=inputs)

        # Synthesize results using the judge
        judged_result = self.judge(
            inputs={"query": inputs.query, "responses": [s1_out.final_answer, s2_out.final_answer]}
        )

        # Extract the final answer
        final_answer = judged_result.final_answer

        # Return structured output
        return NetworkOutput(final_answer=final_answer)


def nested_module_graph() -> Operator[NetworkInput, NetworkOutput]:
    """Creates an instance of the NestedNetwork operator representing a complex nested network structure.

    Returns:
        A strongly-typed NestedNetwork operator instance for pipeline execution.
    """
    return NestedNetwork()


if __name__ == "__main__":
    # Initialize the ember context first
    context = get_ember_context()

    # Quick test invocation using explicit method calls with named parameters.
    network = nested_module_graph()
    test_input = NetworkInput(query="Hello from the new approach")
    test_result = network(inputs=test_input)
    print(f"NestedNetwork final output: {test_result.final_answer}")
