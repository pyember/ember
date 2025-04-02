"""Integration tests for the dependency analyzer.

Validates the core dependency analyzer functionality that powers graph execution.
"""

import pytest
from typing import List, Dict, Any, ClassVar

from ember.core.registry.operator.base.operator_base import Operator, Specification
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.graph.dependency_analyzer import DependencyAnalyzer


class TestOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Simple test operator."""
    
    specification: ClassVar[Specification] = Specification()
    
    def __init__(self, *, name: str) -> None:
        self.name = name
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"Processed by {self.name}"}


def test_simple_dependency_analysis():
    """Test analyzing dependencies in a simple linear graph."""
    # Create a graph
    graph = XCSGraph()
    
    # Add operators
    op1 = TestOperator(name="op1")
    op2 = TestOperator(name="op2")
    op3 = TestOperator(name="op3")
    
    # Add nodes
    node1 = graph.add_node(operator=op1, name="node1")
    node2 = graph.add_node(operator=op2, name="node2")
    node3 = graph.add_node(operator=op3, name="node3")
    
    # Create a linear chain: node1 -> node2 -> node3
    graph.add_edge(node1, node2)
    graph.add_edge(node2, node3)
    
    # Create analyzer
    analyzer = DependencyAnalyzer()
    
    # Analyze direct dependencies
    dependency_map = analyzer.build_dependency_graph(graph)
    
    # Check that dependencies are correct
    assert node3 in dependency_map, f"Node {node3} not in dependency map"
    assert node2 in dependency_map[node3], f"Node {node2} not a dependency of {node3}"
    
    assert node2 in dependency_map, f"Node {node2} not in dependency map"
    assert node1 in dependency_map[node2], f"Node {node1} not a dependency of {node2}"
    
    assert node1 in dependency_map, f"Node {node1} not in dependency map"
    assert len(dependency_map[node1]) == 0, f"Node {node1} should have no dependencies"


def test_complex_dependency_analysis():
    """Test analyzing dependencies in a graph with branches and joins."""
    # Create a graph
    graph = XCSGraph()
    
    # Create a diamond pattern:
    #   A
    #  / \
    # B   C
    #  \ /
    #   D
    
    # Add operators
    op_a = TestOperator(name="A")
    op_b = TestOperator(name="B")
    op_c = TestOperator(name="C")
    op_d = TestOperator(name="D")
    
    # Add nodes
    node_a = graph.add_node(operator=op_a, name="A")
    node_b = graph.add_node(operator=op_b, name="B")
    node_c = graph.add_node(operator=op_c, name="C")
    node_d = graph.add_node(operator=op_d, name="D")
    
    # Create edges
    graph.add_edge(node_a, node_b)
    graph.add_edge(node_a, node_c)
    graph.add_edge(node_b, node_d)
    graph.add_edge(node_c, node_d)
    
    # Create analyzer
    analyzer = DependencyAnalyzer()
    
    # Analyze direct dependencies
    dependency_map = analyzer.build_dependency_graph(graph)
    
    # Check direct dependencies
    assert node_b in dependency_map[node_d], "B should be a direct dependency of D"
    assert node_c in dependency_map[node_d], "C should be a direct dependency of D"
    
    # Check transitive dependencies
    transitive_deps = analyzer.compute_transitive_closure(dependency_map)
    assert node_a in transitive_deps[node_d], "A should be a transitive dependency of D"
    
    # Check execution waves
    waves = analyzer.compute_execution_waves(graph)
    assert len(waves) == 3, f"Should have 3 execution waves, got {len(waves)}"
    
    # Wave 1 should be just A
    assert node_a in waves[0], "A should be in wave 1"
    assert len(waves[0]) == 1, f"Wave 1 should have 1 node, got {len(waves[0])}"
    
    # Wave 2 should be B and C
    assert node_b in waves[1], "B should be in wave 2"
    assert node_c in waves[1], "C should be in wave 2"
    assert len(waves[1]) == 2, f"Wave 2 should have 2 nodes, got {len(waves[1])}"
    
    # Wave 3 should be just D
    assert node_d in waves[2], "D should be in wave 3"
    assert len(waves[2]) == 1, f"Wave 3 should have 1 node, got {len(waves[2])}"