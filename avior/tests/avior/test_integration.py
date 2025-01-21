import pytest
from avior.graph import NoNGraphBuilder, GraphExecutor


def test_full_integration():
    """
    Full integration:
    - Use ENSEMBLE (shorthand), MOST_COMMON, and GET_ANSWER.
    - Provide initial query input, run GraphExecutor and verify final output.

    Precondition: ENSEMBLE, MOST_COMMON, GET_ANSWER registered.
    TODO: Implement once operators are registered.
    """
    graph_def = {
        "ens": {"op": "3:ENSEMBLE:gpt-4o:1.0", "inputs": []},
        "mc": {
            "op": "MOST_COMMON",
            "params": {"model_name": "gpt-4o"},
            "inputs": ["ens"],
        },
        "ga": {
            "op": "GET_ANSWER",
            "params": {"model_name": "gpt-4o"},
            "inputs": ["mc"],
        },
    }

    builder = NoNGraphBuilder()
    graph_data = builder.parse_graph(graph_def)
    executor = GraphExecutor()
    result = executor.execute(graph_data, {"query": "Hello world"})
    # TODO: Assert final result correctness. Possibly final_answer from GET_ANSWER node.
