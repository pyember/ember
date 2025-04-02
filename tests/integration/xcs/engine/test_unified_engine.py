"""Integration tests for the unified XCS execution engine.

Tests the unified execution engine with various execution options,
graph configurations, and scheduler combinations.
"""

import pytest
import time
from typing import Dict, Any, ClassVar, List, Optional

from ember.core.registry.operator.base.operator_base import Operator, Specification
from ember.xcs import jit, execution_options
from ember.xcs.engine.unified_engine import execute_graph
from ember.xcs.graph.xcs_graph import XCSGraph


class SimpleOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Simple operator for testing."""
    
    specification: ClassVar[Specification] = Specification()
    
    def __init__(self, *, value: int = 1, delay: float = 0.01) -> None:
        self.value = value
        self.delay = delay
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs.get("value", 0) + self.value
        time.sleep(self.delay)  # Small delay for testing parallelism
        return {"value": result}


def test_execute_graph_core_functionality():
    """Test the core functionality of execute_graph."""
    # Create a simple graph
    graph = XCSGraph()
    
    # Add operators as nodes
    op1 = SimpleOperator(value=5)
    op2 = SimpleOperator(value=10)
    
    # Define a simple input function that emits the initial value
    def input_fn(**kwargs):
        return {"value": 0}
    
    # Create nodes in the graph
    input_node = graph.add_node(input_fn, name="input")
    node1 = graph.add_node(op1, name="op1")
    node2 = graph.add_node(op2, name="op2")
    
    # Define the execution flow
    graph.add_edge(input_node, node1)
    graph.add_edge(node1, node2)
    
    # Execute the graph with empty inputs (our input_fn doesn't use them)
    result = execute_graph(graph, inputs={})
    
    # Verify the result - node IDs are strings in the result dict
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    assert node2 in result, f"Node {node2} not found in result: {result}"
    assert "value" in result[node2], f"'value' not found in node {node2} result: {result[node2]}"
    assert result[node2]["value"] == 15, f"Expected value 15, got {result[node2]['value']}"