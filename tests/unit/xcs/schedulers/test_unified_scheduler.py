"""Tests for the unified scheduler system."""

import pytest
from typing import Dict, Any, Callable

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.schedulers.unified_scheduler import (
    SequentialScheduler,
    ParallelScheduler,
    WaveScheduler,
    NoOpScheduler,
    TopologicalScheduler
)


class SimpleOperator:
    """Simple test operator for scheduler tests."""
    
    def __init__(self, name: str, value_multiplier: int = 1) -> None:
        self.name = name
        self.value_multiplier = value_multiplier
        self.call_count = 0
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        return {"result": inputs.get("value", 1) * self.value_multiplier}


def create_test_graph() -> XCSGraph:
    """Create a simple test graph for scheduler tests."""
    graph = XCSGraph()
    
    # Create operators
    op1 = SimpleOperator("op1", 2)
    op2 = SimpleOperator("op2", 3)
    op3 = SimpleOperator("op3", 4)
    
    # Add nodes to graph
    node1 = graph.add_node(operator=op1, name="node1")
    node2 = graph.add_node(operator=op2, name="node2")
    node3 = graph.add_node(operator=op3, name="node3")
    
    # Add edges (node1 -> node2 -> node3) with field mappings
    graph.add_edge(node1, node2, {"result": "value"})  # Map op1's result to op2's value input
    graph.add_edge(node2, node3, {"result": "value"})  # Map op2's result to op3's value input
    
    return graph


def test_sequential_scheduler():
    """Test sequential scheduler execution."""
    graph = create_test_graph()
    scheduler = SequentialScheduler()
    
    # Store original nodes for verification
    nodes = {}
    for node_id, node in graph.nodes.items():
        operator = node.operator
        if hasattr(operator, 'name'):
            nodes[operator.name] = node_id
    
    # Prepare and execute
    scheduler.prepare(graph)
    results = scheduler.execute(graph, {"value": 5})
    
    # Find results by examining node operators
    op1_result = None
    op2_result = None
    op3_result = None
    
    for node_id, result in results.items():
        if node_id in graph.nodes:
            op = graph.nodes[node_id].operator
            if op.name == "op1":
                op1_result = result
            elif op.name == "op2":
                op2_result = result
            elif op.name == "op3":
                op3_result = result
    
    # Verify results
    assert op1_result is not None, "op1 result not found"
    assert op2_result is not None, "op2 result not found"
    assert op3_result is not None, "op3 result not found"
    assert op1_result["result"] == 10  # 5 * 2
    assert op2_result["result"] == 30  # 10 * 3
    assert op3_result["result"] == 120  # 30 * 4


def test_topological_scheduler():
    """Test topological scheduler execution."""
    graph = create_test_graph()
    scheduler = TopologicalScheduler()
    
    # Prepare and execute
    scheduler.prepare(graph)
    results = scheduler.execute(graph, {"value": 5})
    
    # Find results by examining node operators
    op1_result = None
    op2_result = None
    op3_result = None
    
    for node_id, result in results.items():
        if node_id in graph.nodes:
            op = graph.nodes[node_id].operator
            if op.name == "op1":
                op1_result = result
            elif op.name == "op2":
                op2_result = result
            elif op.name == "op3":
                op3_result = result
    
    # Verify results
    assert op1_result is not None, "op1 result not found"
    assert op2_result is not None, "op2 result not found"
    assert op3_result is not None, "op3 result not found"
    assert op1_result["result"] == 10  # 5 * 2
    assert op2_result["result"] == 30  # 10 * 3
    assert op3_result["result"] == 120  # 30 * 4


def test_parallel_scheduler():
    """Test parallel scheduler execution."""
    graph = create_test_graph()
    scheduler = ParallelScheduler(max_workers=2)
    
    # Prepare and execute
    scheduler.prepare(graph)
    results = scheduler.execute(graph, {"value": 5})
    
    # Find results by examining node operators
    op1_result = None
    op2_result = None
    op3_result = None
    
    for node_id, result in results.items():
        if node_id in graph.nodes:
            op = graph.nodes[node_id].operator
            if op.name == "op1":
                op1_result = result
            elif op.name == "op2":
                op2_result = result
            elif op.name == "op3":
                op3_result = result
    
    # Verify results
    assert op1_result is not None, "op1 result not found"
    assert op2_result is not None, "op2 result not found"
    assert op3_result is not None, "op3 result not found"
    assert op1_result["result"] == 10  # 5 * 2
    assert op2_result["result"] == 30  # 10 * 3
    assert op3_result["result"] == 120  # 30 * 4


def test_wave_scheduler():
    """Test wave scheduler execution."""
    graph = create_test_graph()
    scheduler = WaveScheduler(max_workers=2)
    
    # Prepare and execute
    scheduler.prepare(graph)
    results = scheduler.execute(graph, {"value": 5})
    
    # Find results by examining node operators
    op1_result = None
    op2_result = None
    op3_result = None
    
    for node_id, result in results.items():
        if node_id in graph.nodes:
            op = graph.nodes[node_id].operator
            if op.name == "op1":
                op1_result = result
            elif op.name == "op2":
                op2_result = result
            elif op.name == "op3":
                op3_result = result
    
    # Verify results
    assert op1_result is not None, "op1 result not found"
    assert op2_result is not None, "op2 result not found"
    assert op3_result is not None, "op3 result not found"
    assert op1_result["result"] == 10  # 5 * 2
    assert op2_result["result"] == 30  # 10 * 3
    assert op3_result["result"] == 120  # 30 * 4


def test_noop_scheduler():
    """Test no-op scheduler execution."""
    graph = create_test_graph()
    scheduler = NoOpScheduler()
    
    # Store references to operators for later verification
    operators = {}
    for node_id, node in graph.nodes.items():
        op = node.operator
        if hasattr(op, 'name'):
            operators[op.name] = op
    
    # Prepare and execute
    scheduler.prepare(graph)
    results = scheduler.execute(graph, {"value": 5})
    
    # Find results by examining node operators
    op1_result = None
    op2_result = None
    op3_result = None
    
    for node_id, result in results.items():
        if node_id in graph.nodes:
            op = graph.nodes[node_id].operator
            if op.name == "op1":
                op1_result = result
            elif op.name == "op2":
                op2_result = result
            elif op.name == "op3":
                op3_result = result
    
    # Verify no execution occurred but results were created
    assert op1_result is not None, "op1 result not found"
    assert op2_result is not None, "op2 result not found"
    assert op3_result is not None, "op3 result not found"
    
    # All nodes should have "_simulated" flag
    assert op1_result["_simulated"] is True
    assert op2_result["_simulated"] is True
    assert op3_result["_simulated"] is True
    
    # Operators should not have been called
    op1 = operators["op1"]
    op2 = operators["op2"]
    op3 = operators["op3"]
    
    assert op1.call_count == 0
    assert op2.call_count == 0
    assert op3.call_count == 0