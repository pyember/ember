"""
Tests for XCS (eXecution Control System) type definitions.
"""

import pytest
from typing import Dict, Any, List, Optional, Union

from ember.core.types.xcs_types import (
    XCSNode,
    XCSGraph,
    XCSPlan,
    XCSNodeAttributes,
    XCSNodeResult,
)


class MockNode:
    """Mock implementation of XCSNode for testing."""

    def __init__(self, node_id: str, operator: Any):
        self.node_id = node_id
        self.operator = operator
        self.inbound_edges: List[str] = []
        self.outbound_edges: List[str] = []
        self.attributes: Dict[str, Any] = {}
        self.captured_outputs: Any = None


class MockGraph:
    """Mock implementation of XCSGraph for testing."""

    def __init__(self):
        self.nodes: Dict[str, MockNode] = {}

    def add_node(self, node_id: str, operator: Any, **attributes: Any) -> None:
        """Add a node to the graph."""
        node = MockNode(node_id=node_id, operator=operator)
        node.attributes = attributes
        self.nodes[node_id] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Nodes {from_node} or {to_node} not found")

        self.nodes[from_node].outbound_edges.append(to_node)
        self.nodes[to_node].inbound_edges.append(from_node)

    def get_node(self, node_id: str) -> MockNode:
        """Get a node by ID."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        return self.nodes[node_id]


class MockPlan:
    """Mock implementation of XCSPlan for testing."""

    def __init__(self, tasks: Dict[str, Any], original_graph: Any):
        self.tasks = tasks
        self.original_graph = original_graph


def test_xcs_node_protocol():
    """Test that MockNode satisfies the XCSNode protocol."""

    def sample_op(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": inputs.get("value", 0) * 2}

    node = MockNode(node_id="test_node", operator=sample_op)

    # Test runtime protocol checking
    assert isinstance(node, XCSNode)

    # Test node attributes
    assert node.node_id == "test_node"
    assert node.operator is sample_op
    assert node.inbound_edges == []
    assert node.outbound_edges == []
    assert node.attributes == {}
    assert node.captured_outputs is None


def test_xcs_graph_protocol():
    """Test that MockGraph satisfies the XCSGraph protocol."""
    graph = MockGraph()

    def op1(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": 42}

    def op2(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": inputs.get("value", 0) * 2}

    # Add nodes and edges
    graph.add_node("node1", op1, name="Source Node")
    graph.add_node("node2", op2, name="Transform Node")
    graph.add_edge("node1", "node2")

    # Test runtime protocol checking
    assert isinstance(graph, XCSGraph)

    # Test graph operations
    assert len(graph.nodes) == 2
    assert "node1" in graph.nodes
    assert "node2" in graph.nodes

    # Test node connections
    assert graph.nodes["node1"].outbound_edges == ["node2"]
    assert graph.nodes["node2"].inbound_edges == ["node1"]

    # Test node attributes
    assert graph.nodes["node1"].attributes == {"name": "Source Node"}
    assert graph.nodes["node2"].attributes == {"name": "Transform Node"}


def test_xcs_plan_protocol():
    """Test that MockPlan satisfies the XCSPlan protocol."""
    graph = MockGraph()
    tasks = {"task1": {"node_id": "node1", "operator": lambda x: x}}
    plan = MockPlan(tasks=tasks, original_graph=graph)

    # Test runtime protocol checking
    assert isinstance(plan, XCSPlan)

    # Test plan attributes
    assert plan.tasks == tasks
    assert plan.original_graph is graph


def test_xcs_node_attributes():
    """Test XCSNodeAttributes TypedDict."""
    # Create a dict that conforms to XCSNodeAttributes
    attrs: XCSNodeAttributes = {
        "name": "Test Node",
        "description": "A test node for unit testing",
        "tags": ["test", "unit"],
        "metadata": {"priority": 1},
    }

    # We can only test that the structure works as expected
    assert attrs["name"] == "Test Node"
    assert attrs["description"] == "A test node for unit testing"
    assert "test" in attrs["tags"]
    assert attrs["metadata"]["priority"] == 1


def test_xcs_node_result():
    """Test XCSNodeResult TypedDict."""
    # Create a dict that conforms to XCSNodeResult
    result: XCSNodeResult = {
        "success": True,
        "result": {"value": 42},
        "execution_time": 0.05,
        "metadata": {"memory_usage": 1024},
    }

    # We can only test that the structure works as expected
    assert result["success"] is True
    assert result["result"]["value"] == 42
    assert result["execution_time"] == 0.05
    assert result["metadata"]["memory_usage"] == 1024

    # Test with error
    error_result: XCSNodeResult = {
        "success": False,
        "result": None,
        "error": "Division by zero",
    }

    assert error_result["success"] is False
    assert error_result["error"] == "Division by zero"
