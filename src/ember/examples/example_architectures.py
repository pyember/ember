from ember.src.ember.xcs.graph_ir.graph_executor import NoNGraphBuilder, NoNGraphData


def multi_model_graph() -> NoNGraphData:
    """Constructs a multi-model NoN graph with multiple model ensembles and a final judge.

    Builds a NoNGraph structure that includes:
        1. An ensemble of 3 "gpt-4o" models.
        2. An ensemble of 3 "gpt-4-turbo" models.
        3. An ensemble of 4 "gemini-1.5-pro" models.
        4. A final judge using a single "gpt-4o" model.

    Returns:
        NoNGraphData: A configured multi-model graph structure encapsulating diverse model inputs.
    """
    graph_config: dict = {
        "openai_ensemble": {
            "op": "E",
            "params": {"model_name": "gpt-4o", "count": 3},
        },
        "openai_turbo": {
            "op": "E",
            "params": {"model_name": "gpt-4-turbo", "count": 3},
        },
        "gemini_1_5": {
            "op": "E",
            "params": {"model_name": "gemini-1.5-pro", "count": 4},
        },
        "final_judge": {
            "op": "JB",
            "params": {"model_name": "gpt-4o", "count": 1},
            "inputs": [
                "openai_ensemble",
                "openai_turbo",
                "gemini_1_5",
            ],
        },
    }
    return NoNGraphBuilder().parse_graph(graph_config=graph_config)


from typing import Dict, Any
from ember.src.ember.registry.operator.core.operator_base import Operator
from ember.src.ember.xcs import non


class SubNetwork(Operator[Dict[str, Any], Dict[str, Any]]):
    """SubNetwork that composes an ensemble with self-refinement.

    This operator first processes inputs through an ensemble of models and subsequently refines
    the output based on the initial ensemble's response.

    Attributes:
        ensemble (non.Ensemble): An ensemble operator with 2 units of "gpt-4o".
        refine (non.SelfRefinement): A self-refinement operator using "gpt-4o".
    """

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
        s1_out: Dict[str, Any] = self.sub1.forward(inputs=inputs)
        s2_out: Dict[str, Any] = self.sub2.forward(inputs=inputs)
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
