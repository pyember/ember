import pytest
from avior.graph import NoNGraphBuilder, NoNGraphData
from avior.registry.operators.operator_registry import OperatorRegistry


def test_nongraph_builder_simple_op():
    """
    Test building a graph with normal operator code/params.
    Assume ENSEMBLE is registered somewhere (in conftest or in a previous test).
    """
    # TODO: Ensure ENSEMBLE is registered here or globally
    graph_def = {
        "node1": {"op": "ENSEMBLE", "params": {"model_name": "gpt-4o"}, "inputs": []}
    }
    builder = NoNGraphBuilder()
    graph_data = builder.parse_graph(graph_def)
    assert isinstance(graph_data, NoNGraphData)
    assert "node1" in graph_data.nodes


def test_nongraph_builder_shorthand():
    """
    Test shorthand parsing: "3:ENSEMBLE:gpt-4o:1.0".
    """
    # Ensure ENSEMBLE is registered
    graph_def = {"ens": {"op": "3:ENSEMBLE:gpt-4o:1.0", "inputs": []}}
    builder = NoNGraphBuilder()
    graph_data = builder.parse_graph(graph_def)
    op = graph_data.nodes["ens"].operator
    assert len(op.lm_modules) == 3


def test_nongraph_builder_unknown_op():
    """
    Unknown operator code should raise ValueError.
    """
    graph_def = {
        "n1": {"op": "UNKNOWN_OP", "params": {"model_name": "gpt-4o"}, "inputs": []}
    }
    builder = NoNGraphBuilder()
    with pytest.raises(ValueError):
        builder.parse_graph(graph_def)


def test_nongraph_builder_mixed_shorthand_and_normal():
    """
    Mixed scenario:
    - One node uses shorthand
    - Another uses normal code and references first node
    TODO: Implement once multiple operators are registered (ENSEMBLE and MOST_COMMON)
    """
    pass
