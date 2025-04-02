"""Tests for the unified execution engine."""

import pytest
from typing import Dict, Any

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine.unified_engine import (
    execute_graph,
    GraphExecutor,
    ExecutionOptions,
    execution_options
)


class TestOperator:
    """Test operator for engine tests."""
    
    def __init__(self, name: str, multiplier: int = 1) -> None:
        self.name = name
        self.multiplier = multiplier
        self.calls = 0
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        return {"value": inputs.get("value", 1) * self.multiplier}


def create_test_graph() -> XCSGraph:
    """Create a simple test graph for engine tests."""
    graph = XCSGraph()
    
    # Add nodes
    double_op = TestOperator("double", 2)
    triple_op = TestOperator("triple", 3)
    
    node1 = graph.add_node(operator=double_op, name="double")
    node2 = graph.add_node(operator=triple_op, name="triple")
    
    # Connect nodes
    graph.add_edge(node1, node2)
    
    return graph


def test_execute_graph():
    """Test basic graph execution."""
    graph = create_test_graph()
    
    # Execute the graph
    results = execute_graph(graph, {"value": 5})
    
    # Find nodes by examining values in the results
    double_result = None
    triple_result = None
    
    for node_id, node_result in results.items():
        if node_result.get("value") == 10:
            double_result = node_result
        elif node_result.get("value") == 30:
            triple_result = node_result
            
    # Verify results
    assert double_result is not None, "Double operator result not found"
    assert triple_result is not None, "Triple operator result not found"
    assert double_result["value"] == 10  # 5 * 2
    assert triple_result["value"] == 30  # 10 * 3


def test_execution_options():
    """Test execution options configuration."""
    # Create options
    options = ExecutionOptions(
        scheduler_type="sequential",
        timeout_seconds=10.0,
        continue_on_error=True
    )
    
    # Check settings
    assert options.scheduler_type == "sequential"
    assert options.timeout_seconds == 10.0
    assert options.continue_on_error is True


def test_context_manager():
    """Test execution options context manager."""
    graph = create_test_graph()
    
    # Execute with context manager
    with execution_options(scheduler="sequential", max_workers=None):
        results = execute_graph(graph, {"value": 5})
    
    # Find nodes by examining values in the results
    double_result = None
    triple_result = None
    
    for node_id, node_result in results.items():
        if node_result.get("value") == 10:
            double_result = node_result
        elif node_result.get("value") == 30:
            triple_result = node_result
            
    # Verify results
    assert double_result is not None, "Double operator result not found"
    assert triple_result is not None, "Triple operator result not found"
    assert double_result["value"] == 10  # 5 * 2
    assert triple_result["value"] == 30  # 10 * 3
    
    # Test nested context
    with execution_options(scheduler="parallel", max_workers=2):
        with execution_options(max_workers=1):  # Inner overrides outer
            results = execute_graph(graph, {"value": 5})
    
    # Find nodes by examining values in the results
    double_result = None
    triple_result = None
    
    for node_id, node_result in results.items():
        if node_result.get("value") == 10:
            double_result = node_result
        elif node_result.get("value") == 30:
            triple_result = node_result
            
    # Verify results
    assert double_result is not None, "Double operator result not found"
    assert triple_result is not None, "Triple operator result not found"
    assert double_result["value"] == 10  # 5 * 2
    assert triple_result["value"] == 30  # 10 * 3