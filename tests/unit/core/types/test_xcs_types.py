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
    """
    Mock implementation of XCSNode for testing.
    
    This class satisfies the XCSNode protocol with a simple in-memory implementation
    intended for testing the type system and runtime behavior.
    """

    def __init__(self, node_id: str, operator: Any):
        """
        Initialize a new mock node.
        
        Args:
            node_id: Unique identifier for this node
            operator: Function to execute when this node is run
        """
        self.node_id = node_id  # Unique identifier
        self.operator = operator  # Callable operation
        self.inbound_edges: List[str] = []  # Nodes that feed into this one
        self.outbound_edges: List[str] = []  # Nodes this one feeds into
        self.attributes: XCSNodeAttributes = {}  # Node metadata
        self.captured_outputs: Optional[Dict[str, Any]] = None  # Latest execution result


class MockGraph:
    """
    Mock implementation of XCSGraph for testing.
    
    This class provides a simple in-memory implementation of the XCSGraph protocol
    for unit testing purposes. It maintains a dictionary of nodes and implements 
    the required methods for adding nodes, creating edges, and retrieving nodes.
    """

    def __init__(self):
        """Initialize an empty graph with no nodes or edges."""
        self.nodes: Dict[str, MockNode] = {}

    def add_node(self, node_id: str, operator: Any, **attributes: Any) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            operator: Callable that will be executed when this node runs
            **attributes: Additional attributes to attach to the node
            
        Raises:
            ValueError: If node_id already exists in the graph
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in the graph")
            
        node = MockNode(node_id=node_id, operator=operator)
        node.attributes = attributes
        self.nodes[node_id] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add a directed edge between nodes.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            
        Raises:
            ValueError: If either node doesn't exist in the graph
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Nodes {from_node} or {to_node} not found")

        self.nodes[from_node].outbound_edges.append(to_node)
        self.nodes[to_node].inbound_edges.append(from_node)

    def get_node(self, node_id: str) -> MockNode:
        """
        Get a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            The requested node
        
        Raises:
            KeyError: If node with given ID doesn't exist
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        return self.nodes[node_id]


class MockPlan:
    """
    Mock implementation of XCSPlan for testing.
    
    This class provides a simple in-memory implementation of the XCSPlan protocol
    for unit testing purposes. It maintains task definitions and references to
    the original computation graph.
    """

    def __init__(self, tasks: Dict[str, Any], original_graph: Any):
        """
        Initialize a new execution plan.
        
        Args:
            tasks: Dictionary mapping task IDs to task definitions
            original_graph: The computation graph this plan was derived from
        """
        self.tasks = tasks
        self.original_graph = original_graph
        
    def get_execution_order(self) -> List[str]:
        """
        Get the topologically sorted execution order of tasks.
        
        This implementation simply returns the tasks in arbitrary order
        (no dependency sorting), suitable only for testing purposes.
        
        Returns:
            List of task IDs in a valid execution order
        """
        # Simple implementation - return tasks in arbitrary order
        return list(self.tasks.keys())


def test_xcs_node_protocol():
    """
    Test that MockNode satisfies the XCSNode protocol.
    
    This test verifies two important aspects:
    1. The MockNode class is recognized by isinstance() as implementing XCSNode
    2. The required attributes and their types are properly implemented
    
    Protocol conformance is essential for runtime type safety in a system
    with polymorphic components.
    """
    # Define a simple operator function that doubles numeric inputs
    def sample_op(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Double the 'value' field from the input."""
        return {"result": inputs.get("value", 0) * 2}

    # Create a test node with the sample operator
    node = MockNode(node_id="test_node", operator=sample_op)

    # Verify runtime protocol conformance
    assert isinstance(node, XCSNode), "MockNode should satisfy the XCSNode protocol"

    # Verify all required attributes exist with correct types and values
    assert node.node_id == "test_node", "node_id should match the constructor value"
    assert node.operator is sample_op, "operator should reference the provided function"
    assert node.inbound_edges == [], "New node should have empty inbound edges"
    assert node.outbound_edges == [], "New node should have empty outbound edges"
    assert node.attributes == {}, "New node should have empty attributes"
    assert node.captured_outputs is None, "New node should have no captured outputs"


def test_xcs_graph_protocol():
    """
    Test that MockGraph satisfies the XCSGraph protocol.
    
    This test verifies:
    1. MockGraph implements the XCSGraph protocol
    2. The graph operations (add_node, add_edge, get_node) work correctly
    3. The graph properly maintains node connectivity information
    4. Node attributes are properly stored and retrieved
    
    These verifications ensure that the graph can be correctly constructed
    and later traversed during execution planning.
    """
    # Create an empty graph
    graph = MockGraph()

    # Define two simple operators for testing
    def op1(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Source node that always outputs the value 42."""
        return {"value": 42}

    def op2(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform node that doubles the input value."""
        return {"result": inputs.get("value", 0) * 2}

    # Build a simple two-node graph with metadata
    graph.add_node("node1", op1, name="Source Node", description="Produces initial value")
    graph.add_node("node2", op2, name="Transform Node", description="Doubles input value")
    graph.add_edge("node1", "node2")

    # Verify protocol conformance
    assert isinstance(graph, XCSGraph), "MockGraph should satisfy the XCSGraph protocol"

    # Verify graph structure
    assert len(graph.nodes) == 2, "Graph should contain exactly two nodes"
    assert "node1" in graph.nodes, "Graph should contain 'node1'"
    assert "node2" in graph.nodes, "Graph should contain 'node2'"

    # Verify node connectivity
    assert graph.nodes["node1"].outbound_edges == ["node2"], "node1 should connect to node2"
    assert graph.nodes["node2"].inbound_edges == ["node1"], "node2 should receive from node1"
    assert not graph.nodes["node1"].inbound_edges, "node1 should have no inputs"
    assert not graph.nodes["node2"].outbound_edges, "node2 should have no outputs"

    # Verify node attributes were stored correctly
    assert graph.nodes["node1"].attributes["name"] == "Source Node"
    assert graph.nodes["node1"].attributes["description"] == "Produces initial value"
    assert graph.nodes["node2"].attributes["name"] == "Transform Node"
    assert graph.nodes["node2"].attributes["description"] == "Doubles input value"


def test_xcs_plan_protocol():
    """
    Test that MockPlan satisfies the XCSPlan protocol.
    
    This test verifies:
    1. MockPlan implements the XCSPlan protocol
    2. The plan properly stores task definitions
    3. The plan maintains a reference to its source graph
    4. The execution order method returns all tasks
    
    These verifications ensure that the execution plan can be correctly
    constructed from a graph and later used to drive execution.
    """
    # Create a simple graph for the plan
    graph = MockGraph()
    
    # Define a sample task with minimal required fields
    task_definition = {
        "node_id": "node1",
        "dependencies": [],
        "priority": 1,
        "timeout_seconds": 60.0,
        "retry_policy": "none",
        "max_retries": 0
    }
    
    # Create a plan with a single task
    tasks = {"task1": task_definition}
    plan = MockPlan(tasks=tasks, original_graph=graph)

    # Verify protocol conformance
    assert isinstance(plan, XCSPlan), "MockPlan should satisfy the XCSPlan protocol"

    # Verify plan structure
    assert plan.tasks == tasks, "Plan should store the task definitions"
    assert plan.original_graph is graph, "Plan should reference the source graph"
    
    # Verify execution order contains all tasks
    execution_order = plan.get_execution_order()
    assert len(execution_order) == 1, "Execution order should contain exactly one task"
    assert "task1" in execution_order, "Execution order should include 'task1'"


def test_xcs_node_attributes():
    """
    Test XCSNodeAttributes TypedDict structure and validation.
    
    This test verifies:
    1. The XCSNodeAttributes TypedDict accepts the expected fields
    2. Field values are correctly stored and accessed
    3. The structure works with normal dictionary operations
    
    Type checking is performed by static analysis tools like mypy, not at runtime,
    so we can only verify proper dictionary behavior here.
    """
    # Create a complete node attributes dictionary with all optional fields
    attrs: XCSNodeAttributes = {
        "name": "Test Node",
        "description": "A test node for unit testing",
        "tags": ["test", "unit", "example"],
        "metadata": {
            "source_file": "/path/to/file.py",
            "source_line": 42,
            "author": "Test Author",
            "version": "1.0.0",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
            "description": "Detailed node metadata",
            "custom_data": {"importance": "high", "category": "test"}
        },
    }

    # Verify dictionary access works as expected
    assert attrs["name"] == "Test Node", "Name field should be accessible"
    assert attrs["description"] == "A test node for unit testing", "Description field should be accessible"
    assert len(attrs["tags"]) == 3, "Tags field should contain 3 items"
    assert "test" in attrs["tags"], "Tags field should contain 'test'"
    
    # Verify nested metadata
    assert attrs["metadata"]["source_file"] == "/path/to/file.py"
    assert attrs["metadata"]["source_line"] == 42
    assert attrs["metadata"]["custom_data"]["importance"] == "high"
    
    # Verify dictionary operations
    keys = set(attrs.keys())
    assert keys == {"name", "description", "tags", "metadata"}, "Dictionary keys should match expected fields"


def test_xcs_node_result():
    """
    Test XCSNodeResult TypedDict structure and behavior.
    
    This test verifies:
    1. The XCSNodeResult TypedDict correctly handles both success and error cases
    2. Required and optional fields are properly supported
    3. Nested metadata structure works as expected
    
    These verifications ensure the result structure can properly express
    the outcome of node execution in all relevant scenarios.
    """
    # Create a successful execution result with all fields
    success_result: XCSNodeResult[Dict[str, int]] = {
        "success": True,
        "result": {"value": 42, "extra": 123},
        "execution_time": 0.05,
        "metadata": {
            "start_time": 1672531200.0,  # 2023-01-01T00:00:00Z
            "end_time": 1672531200.05,   # 0.05 seconds later
            "execution_time": 0.05,       # Actual execution time
            "memory_usage": 1024,         # 1KB memory used
            "cpu_usage": 12.5,            # 12.5% CPU used
            "device": "cpu:0",            # Executed on first CPU
            "custom_data": {              # Custom metrics
                "cache_hits": 3,
                "cache_misses": 0
            }
        },
    }

    # Verify successful result fields
    assert success_result["success"] is True, "Success flag should be True"
    assert success_result["result"]["value"] == 42, "Result should contain expected output value"
    assert success_result["execution_time"] == 0.05, "Execution time should be recorded"
    assert success_result["metadata"]["memory_usage"] == 1024, "Memory usage should be recorded"
    assert success_result["metadata"]["custom_data"]["cache_hits"] == 3, "Custom metrics should be accessible"

    # Create an error result with just the required fields
    error_result: XCSNodeResult[Dict[str, Any]] = {
        "success": False,
        "result": None,
        "error": "Division by zero",
    }

    # Verify error result fields
    assert error_result["success"] is False, "Success flag should be False for errors"
    assert error_result["result"] is None, "Result should be None on error"
    assert error_result["error"] == "Division by zero", "Error message should be provided"
    assert "execution_time" not in error_result, "Optional fields can be omitted"
    
    # Verify type safety with generic type parameter (checked by mypy, not runtime)
