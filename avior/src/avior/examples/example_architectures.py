
from avior.core.graph_executor import NoNGraphBuilder, NoNGraphData

def multi_model_graph() -> NoNGraphData:
    """Builds a multi-model NoN graph with multiple model ensembles and a final judge.

    This function constructs a NoNGraph structure with:
      1. An ensemble of 3 "gpt-4o" models.
      2. An ensemble of 3 "gpt-4-turbo" models.
      3. An ensemble of 4 "gemini-1.5-pro" models.
      4. A final judge using "gpt-4o" with count=1.

    Returns:
        NoNGraph: A multi-model graph structure with diverse model inputs.
    """
    graph_config = {
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
    return NoNGraphBuilder().parse_graph(graph_config,)


from typing import Dict, Any
from avior.registry.operator.operator_base import Operator
from avior.registry import non 


class SubNetwork(Operator[Dict[str, Any], Dict[str, Any]]):
    """
    A subnetwork that uses an ensemble + self-refinement.
    We'll keep it untyped for demonstration => T_in=Dict, T_out=Dict
    """

    def __init__(self):
        super().__init__()
        self.ensemble = non.Ensemble(num_units=2, model_name="gpt-4o")
        self.refine = non.SelfRefinement(model_name="gpt-4o")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ens_out = self.ensemble(inputs)  # calls EnsembleInputs(...) under the hood
        ref_out = self.refine({"query": inputs.get("query", ""), "responses": [ens_out]})
        return ref_out


class NestedNetwork(Operator[Dict[str, Any], Dict[str, Any]]):
    def __init__(self):
        super().__init__()
        self.sub1 = SubNetwork()
        self.sub2 = SubNetwork()
        self.judge = non.Judge(model_name="gpt-4o")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        s1_out = self.sub1(inputs)
        s2_out = self.sub2(inputs)
        judge_in = {
            "query": inputs.get("query", ""),
            "responses": [
                s1_out.get("final_answer", ""),
                s2_out.get("final_answer", ""),
            ]
        }
        judged = self.judge(judge_in)
        return judged


def nested_module_graph() -> Operator:
    """
    Creates and returns an instance of the NestedNetwork operator
    representing a complex nested network structure.
    """
    return NestedNetwork()


if __name__ == "__main__":
    # Quick test
    network = nested_module_graph()
    result = network({"query": "Hello from the new approach"})
    print("NestedNetwork final output:", result)
