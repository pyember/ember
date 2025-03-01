"""Unit tests for XCSPlan and compile_graph functionality.

This module verifies the integrity of compiling an XCSGraph into an XCSPlan,
ensuring correct task properties, immutability, and robust error handling in
accordance with the Google Python Style Guide.
"""

from typing import Any, Dict

import pytest

from ember.xcs.engine.xcs_engine import XCSPlan, compile_graph
from ember.xcs.graph.xcs_graph import XCSGraph


def dummy_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Echo operator that returns the provided inputs.

    Args:
        inputs (Dict[str, Any]): A dictionary of input parameters.

    Returns:
        Dict[str, Any]: The same dictionary of inputs.
    """
    return inputs


def test_compile_graph() -> None:
    """Test the successful compilation of a simple XCSGraph into an XCSPlan.

    This test constructs a graph with two nodes and one directed edge from 'node1'
    to 'node2'. It then compiles the graph into an XCSPlan and verifies that:
      - The result is an instance of XCSPlan.
      - The plan contains exactly two tasks.
      - Each task possesses the correct node identifier, operator, and inbound
        dependency configuration.
      - The original graph is correctly preserved within the plan.

    Raises:
        AssertionError: If any property of the compiled plan does not match expectations.
    """
    graph: XCSGraph = XCSGraph()
    # Add nodes using named method invocation for clarity.
    graph.add_node(operator=dummy_operator, node_id="node1")
    graph.add_node(operator=dummy_operator, node_id="node2")
    graph.add_edge(from_id="node1", to_id="node2")

    plan: XCSPlan = compile_graph(graph=graph)
    assert isinstance(plan, XCSPlan), "Compiled plan is not an instance of XCSPlan."
    assert len(plan.tasks) == 2, "Expected 2 tasks in the plan, got {}.".format(
        len(plan.tasks)
    )

    task1 = plan.tasks["node1"]
    task2 = plan.tasks["node2"]
    assert (
        task1.node_id == "node1"
    ), f"Task node_id is '{task1.node_id}', expected 'node1'."
    assert (
        task1.operator == dummy_operator
    ), "Task operator does not match the expected dummy_operator."
    assert task1.inbound_nodes == [], "Task 'node1' should have no inbound nodes."
    assert task2.inbound_nodes == [
        "node1"
    ], f"Task 'node2' inbound_nodes are {task2.inbound_nodes}, expected ['node1']."
    assert (
        plan.original_graph is graph
    ), "The original graph in the plan does not match the input graph."


def test_compile_graph_duplicate_task() -> None:
    """Test that compiling a graph with duplicate task identifiers raises a ValueError.

    This test simulates a condition where duplicate task IDs exist by manually inserting
    a duplicate node reference into the graph's internal collection before compilation.

    Raises:
        ValueError: If compilation does not raise an error for duplicate task IDs.
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="dup")
    # Introduce a duplicate task by assigning an existing node to a new key.
    graph.nodes["dup2"] = graph.nodes["dup"]

    with pytest.raises(ValueError, match=r"Task 'dup' already exists"):
        _ = compile_graph(graph=graph)


def test_plan_immutability() -> None:
    """Test that an XCSPlan instance remains immutable after creation.

    Attempts to modify the plan's tasks attribute should raise an AttributeError,
    ensuring that the internal state of the plan is preserved post-creation.

    Raises:
        AttributeError: If the tasks attribute can be reassigned without error.
    """
    graph: XCSGraph = XCSGraph()
    # Add a node without explicitly specifying a node_id to test default behavior.
    graph.add_node(operator=dummy_operator)
    plan: XCSPlan = compile_graph(graph=graph)

    with pytest.raises(AttributeError):
        # Attempting to modify an immutable attribute should result in an error.
        plan.tasks = {}  # type: ignore
