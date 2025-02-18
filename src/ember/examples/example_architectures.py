from typing import Dict, Any
from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.core import non
from src.ember.core.registry.prompt_signature.signatures import Signature


class SubNetwork(Operator[Dict[str, Any], Dict[str, Any]]):
    """SubNetwork that composes an ensemble with self-refinement.

    This operator first processes inputs through an ensemble of models and subsequently refines
    the output based on the initial ensemble's response.

    Attributes:
        ensemble (non.Ensemble): An ensemble operator with 2 units of "gpt-4o".
        refine (non.SelfRefinement): A self-refinement operator using "gpt-4o".
    """
    signature: Signature = Signature(input_model=None)
    ensemble: non.Ensemble
    refine: non.SelfRefinement

    def __init__(self) -> None:
        """Initializes the SubNetwork with a specified ensemble and self-refinement components."""
        super().__init__()
        self.ensemble = non.Ensemble(num_units=2, model_name="gpt-4o")
        self.refine = non.SelfRefinement(model_name="gpt-4o")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the input through the ensemble and applies self-refinement.

        Args:
            inputs (Dict[str, Any]): The input dictionary containing at least the key 'query'.

        Returns:
            Dict[str, Any]: The refined output produced by the self-refinement process.
        """
        ens_out: Dict[str, Any] = self.ensemble.forward(inputs=inputs)
        refinement_input: Dict[str, Any] = {
            "query": inputs.get("query", ""),
            "responses": [ens_out],
        }
        ref_out: Dict[str, Any] = self.refine.forward(inputs=refinement_input)
        return ref_out


class NestedNetwork(Operator[Dict[str, Any], Dict[str, Any]]):
    """Nested network that aggregates results from multiple sub-networks and applies a final judgment.

    This operator executes two subnetwork branches and uses a judge operator to synthesize the outputs.

    Attributes:
        sub1 (SubNetwork): The first sub-network instance.
        sub2 (SubNetwork): The second sub-network instance.
        judge (non.Judge): A judge operator using "gpt-4o".
    """
    signature: Signature = Signature(input_model=None)
    sub1: SubNetwork
    sub2: SubNetwork
    judge: non.Judge

    def __init__(self) -> None:
        """Initializes the NestedNetwork with two SubNetwork instances and a final Judge operator."""
        super().__init__()
        self.sub1 = SubNetwork()
        self.sub2 = SubNetwork()
        self.judge = non.Judge(model_name="gpt-4o")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the nested network by processing inputs through sub-networks and aggregating responses.

        Args:
            inputs (Dict[str, Any]): The input data containing the query.

        Returns:
            Dict[str, Any]: The final judged output after aggregating sub-network responses.
        """
        s1_out: Dict[str, Any] = self.sub1(inputs=inputs)
        s2_out: Dict[str, Any] = self.sub2(inputs=inputs)
        judge_input: Dict[str, Any] = {
            "query": inputs.get("query", ""),
            "responses": [
                s1_out.get("final_answer", ""),
                s2_out.get("final_answer", ""),
            ],
        }
        judged: Dict[str, Any] = self.judge.forward(inputs=judge_input)
        return judged


def nested_module_graph() -> Operator:
    """Creates an instance of the NestedNetwork operator representing a complex nested network structure.

    Returns:
        Operator: An instance of NestedNetwork for pipeline execution.
    """
    return NestedNetwork()


if __name__ == "__main__":
    # Quick test invocation using explicit method calls with named parameters.
    network: Operator = nested_module_graph()
    test_input: Dict[str, Any] = {"query": "Hello from the new approach"}
    test_result: Dict[str, Any] = network.forward(inputs=test_input)
    print("NestedNetwork final output:", test_result)
