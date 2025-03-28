"""Unit tests for XCSGraph functionality.

This module tests the functionality of the XCSGraph class, verifying:
    - Proper management of nodes and edges,
    - Accurate topological sorting,
    - Detection of cycles, and
    - Correct merging of graph instances with namespace handling.
"""

from typing import Any, Dict, List

import pytest

from ember.xcs.engine.xcs_engine import execute_graph, TopologicalScheduler
from ember.xcs.graph.xcs_graph import XCSGraph, XCSEdge, merge_xcs_graphs


def dummy_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Simulated operator that returns the provided inputs.

    Args:
        inputs (Dict[str, Any]): A dictionary of input values.

    Returns:
        Dict[str, Any]: The same dictionary of input values.
    """
    return inputs


def test_add_node_and_edge() -> None:
    """Verify that nodes and edges are correctly added to an XCSGraph.

    This test creates a graph with two nodes connected by an edge, then asserts that:
      - Both nodes are present in the graph's registry.
      - The outbound edge from the first node and inbound edge to the second node are set correctly.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="node1")
    graph.add_node(operator=dummy_operator, node_id="node2")
    graph.add_edge(from_id="node1", to_id="node2")
    assert "node1" in graph.nodes, "Expected 'node1' to be present in the graph."
    assert "node2" in graph.nodes, "Expected 'node2' to be present in the graph."
    assert graph.nodes["node1"].outbound_edges == [
        "node2"
    ], "Expected 'node1' to have an outbound edge to 'node2'."
    assert graph.nodes["node2"].inbound_edges == [
        "node1"
    ], "Expected 'node2' to have an inbound edge from 'node1'."


def test_duplicate_node_id_error() -> None:
    """Ensure that adding a duplicate node ID raises a ValueError.

    Attempts to add a node with an identifier that already exists in the graph should raise a ValueError,
    enforcing the uniqueness of node IDs.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="dup")
    with pytest.raises(ValueError, match="Node with ID 'dup' already exists."):
        graph.add_node(operator=dummy_operator, node_id="dup")


def test_topological_sort_linear() -> None:
    """Validate topological sorting on a linear graph.

    Constructs a simple linear graph (A -> B -> C) and ensures that the topological sort
    yields the order ['A', 'B', 'C'].

    Raises:
        AssertionError: If the sorted order does not match the expected sequence.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="A")
    graph.add_node(operator=dummy_operator, node_id="B")
    graph.add_node(operator=dummy_operator, node_id="C")
    graph.add_edge(from_id="A", to_id="B")
    graph.add_edge(from_id="B", to_id="C")
    order: List[str] = graph.topological_sort()
    assert order == [
        "A",
        "B",
        "C",
    ], "Topological sort should yield ['A', 'B', 'C'] for a linear graph."


def test_topological_sort_diamond() -> None:
    """Verify topological sorting on a diamond-shaped graph.

    Constructs a diamond-shaped graph:
          A
         / \\
        B   C
         \\ /
          D
    The test asserts that 'A' is sorted first, 'D' is sorted last, and 'B' and 'C' appear in between.

    Raises:
        AssertionError: If the sorted order does not comply with the diamond topology constraints.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="A")
    graph.add_node(operator=dummy_operator, node_id="B")
    graph.add_node(operator=dummy_operator, node_id="C")
    graph.add_node(operator=dummy_operator, node_id="D")
    graph.add_edge(from_id="A", to_id="B")
    graph.add_edge(from_id="A", to_id="C")
    graph.add_edge(from_id="B", to_id="D")
    graph.add_edge(from_id="C", to_id="D")
    order: List[str] = graph.topological_sort()
    assert order[0] == "A", "The first node should be 'A'."
    assert order[-1] == "D", "The last node should be 'D'."
    assert set(order[1:-1]) == {"B", "C"}, "Intermediate nodes should be 'B' and 'C'."


def test_cycle_detection() -> None:
    """Test that a cycle in the graph triggers a ValueError during topological sorting.

    Constructs a cyclic graph with two nodes forming a loop. The topological_sort
    method is expected to detect the cycle and raise a ValueError.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="1")
    graph.add_node(operator=dummy_operator, node_id="2")
    graph.add_edge(from_id="1", to_id="2")
    graph.add_edge(from_id="2", to_id="1")
    with pytest.raises(ValueError, match="Graph contains a cycle"):
        graph.topological_sort()


def test_merge_xcs_graphs_namespace() -> None:
    """Test merging two graphs with namespace prefixing.

    Verifies that merging a base graph with an additional graph results in node IDs
    from the additional graph being correctly prefixed with the given namespace to avoid conflicts.
    """
    base_graph: XCSGraph = XCSGraph()
    additional_graph: XCSGraph = XCSGraph()
    base_graph.add_node(operator=dummy_operator, node_id="base1")
    additional_graph.add_node(operator=dummy_operator, node_id="add1")
    merged_graph: XCSGraph = merge_xcs_graphs(
        base=base_graph, additional=additional_graph, namespace="Test"
    )
    assert (
        "base1" in merged_graph.nodes
    ), "Merged graph must contain 'base1' from the base graph."
    assert (
        "Test_add1" in merged_graph.nodes
    ), "Merged graph must contain 'Test_add1' from the namespaced additional graph."


def test_merge_with_duplicates() -> None:
    """Ensure proper renaming when merging graphs with duplicate node IDs.

    When both the base and additional graphs contain a node with the identical ID,
    the additional node should be renamed with the provided namespace to guarantee uniqueness.

    Raises:
        AssertionError: If the merged graph does not include the correctly renamed node.
    """
    base_graph: XCSGraph = XCSGraph()
    additional_graph: XCSGraph = XCSGraph()
    base_graph.add_node(operator=dummy_operator, node_id="shared")
    additional_graph.add_node(operator=dummy_operator, node_id="shared")
    merged_graph: XCSGraph = merge_xcs_graphs(
        base=base_graph, additional=additional_graph, namespace="Ns"
    )
    assert (
        "shared" in merged_graph.nodes
    ), "Merged graph should contain 'shared' from the base graph."
    assert any(
        node_id.startswith("Ns_shared") for node_id in merged_graph.nodes
    ), "Merged graph must include a namespaced version of the duplicate node from the additional graph."


def test_field_level_mapping() -> None:
    """Test field-level mappings between nodes.

    Verifies that field mappings work correctly to route specific outputs from
    one node to specific inputs of another node.
    """
    # Create graph with three nodes
    graph: XCSGraph = XCSGraph()

    # Define custom operators that will verify correct field mapping
    def producer_op(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Produces multiple outputs."""
        return {"value1": 10, "value2": 20, "metadata": "test"}

    def consumer1_op(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Consumes value1 and doubles it."""
        assert "value1" in inputs, "Expected 'value1' in inputs"
        assert "value2" not in inputs, "Should not receive 'value2'"
        return {"result": inputs["value1"] * 2}

    def consumer2_op(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Consumes value2 and triples it."""
        assert "value2" in inputs, "Expected 'value2' in inputs"
        assert "value1" not in inputs, "Should not receive 'value1'"
        return {"result": inputs["value2"] * 3}

    # Add nodes to graph
    producer_id = graph.add_node(operator=producer_op, node_id="producer")
    consumer1_id = graph.add_node(operator=consumer1_op, node_id="consumer1")
    consumer2_id = graph.add_node(operator=consumer2_op, node_id="consumer2")

    # Add edges with field mappings
    graph.add_edge(
        from_id=producer_id,
        to_id=consumer1_id,
        field_mappings={"value1": "value1"},  # Map only value1
    )

    graph.add_edge(
        from_id=producer_id,
        to_id=consumer2_id,
        field_mappings={"value2": "value2"},  # Map only value2
    )

    # Verify edges and field mappings
    edge_key1 = f"{producer_id}_{consumer1_id}"
    edge_key2 = f"{producer_id}_{consumer2_id}"

    assert edge_key1 in graph.edges, "Edge from producer to consumer1 should exist"
    assert edge_key2 in graph.edges, "Edge from producer to consumer2 should exist"

    assert graph.edges[edge_key1].field_mappings == {"value1": "value1"}
    assert graph.edges[edge_key2].field_mappings == {"value2": "value2"}

    # Prepare inputs for nodes
    results = {"producer": {"value1": 10, "value2": 20, "metadata": "test"}}

    # Test prepare_node_inputs
    consumer1_inputs = graph.prepare_node_inputs("consumer1", results)
    consumer2_inputs = graph.prepare_node_inputs("consumer2", results)

    # Verify only the correct fields were passed
    assert consumer1_inputs == {"value1": 10}
    assert consumer2_inputs == {"value2": 20}

    # Execute operators directly to verify the assertions in the operator functions
    consumer1_result = consumer1_op(inputs=consumer1_inputs)
    consumer2_result = consumer2_op(inputs=consumer2_inputs)

    # Check results
    assert consumer1_result == {"result": 20}  # value1 * 2
    assert consumer2_result == {"result": 60}  # value2 * 3


def test_execute_graph_with_field_mapping() -> None:
    """Test end-to-end graph execution with field mappings.

    Verifies that field mappings are correctly applied during actual graph execution,
    not just in preparation functions.
    """
    # Create graph with three nodes
    graph: XCSGraph = XCSGraph()

    # Define custom operators with clear, trace-able behavior
    def split_op(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Splits input into two outputs."""
        input_value = inputs.get("input", 0)
        return {"double": input_value * 2, "triple": input_value * 3}

    def path1_op(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Verifies it receives only 'double' input and adds 5."""
        assert "double" in inputs, "Expected 'double' input"
        assert "triple" not in inputs, "Should not receive 'triple' input"
        return {"path1_result": inputs["double"] + 5}

    def path2_op(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Verifies it receives only 'triple' input and subtracts 2."""
        assert "triple" in inputs, "Expected 'triple' input"
        assert "double" not in inputs, "Should not receive 'double' input"
        return {"path2_result": inputs["triple"] - 2}

    def merge_op(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Merges results from both paths."""
        assert "path1" in inputs, "Expected 'path1' input"
        assert "path2" in inputs, "Expected 'path2' input"
        return {"final": inputs["path1"] + inputs["path2"]}

    # Add nodes to graph
    split_id = graph.add_node(operator=split_op, node_id="split")
    path1_id = graph.add_node(operator=path1_op, node_id="path1")
    path2_id = graph.add_node(operator=path2_op, node_id="path2")
    merge_id = graph.add_node(operator=merge_op, node_id="merge")

    # Add edges with field mappings
    graph.add_edge(
        from_id=split_id, to_id=path1_id, field_mappings={"double": "double"}
    )

    graph.add_edge(
        from_id=split_id, to_id=path2_id, field_mappings={"triple": "triple"}
    )

    graph.add_edge(
        from_id=path1_id, to_id=merge_id, field_mappings={"path1_result": "path1"}
    )

    graph.add_edge(
        from_id=path2_id, to_id=merge_id, field_mappings={"path2_result": "path2"}
    )

    # Execute the graph with input value 10
    results = execute_graph(
        graph=graph, global_input={"input": 10}, scheduler=TopologicalScheduler()
    )

    # Verify the output of each node
    assert "split" in results
    assert "path1" in results
    assert "path2" in results
    assert "merge" in results

    # Verify split output
    assert results["split"] == {"double": 20, "triple": 30}

    # Verify path1 and path2 outputs
    assert results["path1"] == {"path1_result": 25}  # double (20) + 5
    assert results["path2"] == {"path2_result": 28}  # triple (30) - 2

    # Verify final merged output
    assert results["merge"] == {"final": 53}  # path1 (25) + path2 (28)
