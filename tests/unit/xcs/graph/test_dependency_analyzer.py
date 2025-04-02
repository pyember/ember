"""Unit tests for the XCS graph dependency analyzer.

Tests the dependency analysis functionality for XCS computation graphs,
including topological sorting, transitive closure calculation, and execution
wave computation for parallel scheduling.
"""

import pytest

from ember.xcs import XCSGraph, DependencyAnalyzer


class TestDependencyAnalyzer:
    """Test suite for the DependencyAnalyzer class."""

    def test_basic_dependency_analysis(self) -> None:
        """Test analysis of simple linear dependencies."""
        # Create a simple graph with linear dependencies
        graph = XCSGraph()
        graph.add_node(lambda x: x, "node1")
        graph.add_node(lambda x: x, "node2")
        graph.add_node(lambda x: x, "node3")
        
        # Set up dependencies: node1 -> node2 -> node3
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", "node3")
        
        # Analyze dependencies
        analyzer = DependencyAnalyzer()
        deps = analyzer.analyze(graph)
        
        # Verify expected dependencies
        assert deps["node1"] == set()
        assert deps["node2"] == {"node1"}
        assert deps["node3"] == {"node1", "node2"}

    def test_transitive_closure(self) -> None:
        """Test computation of transitive closure."""
        # Direct dependencies
        direct_deps = {
            "A": {"B", "C"},
            "B": {"D"},
            "C": {"D", "E"},
            "D": {"F"},
            "E": set(),
            "F": set(),
        }
        
        # Analyze with transitive closure
        analyzer = DependencyAnalyzer()
        all_deps = analyzer.compute_transitive_closure(direct_deps)
        
        # Verify expected transitive dependencies
        assert all_deps["A"] == {"B", "C", "D", "E", "F"}
        assert all_deps["B"] == {"D", "F"}
        assert all_deps["C"] == {"D", "E", "F"}
        assert all_deps["D"] == {"F"}
        assert all_deps["E"] == set()
        assert all_deps["F"] == set()

    def test_topological_sort(self) -> None:
        """Test topological sorting of graph nodes."""
        # Create a DAG
        graph = XCSGraph()
        graph.add_node(lambda x: x, "A")
        graph.add_node(lambda x: x, "B")
        graph.add_node(lambda x: x, "C")
        graph.add_node(lambda x: x, "D")
        graph.add_node(lambda x: x, "E")
        
        # Set up dependencies
        # A -> B -> D
        # A -> C -> D
        # C -> E
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        graph.add_edge("C", "D")
        graph.add_edge("C", "E")
        
        # Get topological ordering
        analyzer = DependencyAnalyzer()
        order = analyzer.topological_sort(graph)
        
        # Verify topological properties
        # 1. A comes before B and C
        # 2. B comes before D
        # 3. C comes before D and E
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")
        assert order.index("C") < order.index("E")

    def test_topological_sort_with_cycle(self) -> None:
        """Test topological sorting fails with cycles."""
        # Create a cyclic graph
        graph = XCSGraph()
        graph.add_node(lambda x: x, "A")
        graph.add_node(lambda x: x, "B")
        graph.add_node(lambda x: x, "C")
        
        # Set up cycle: A -> B -> C -> A
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        
        # Verify cycle detection
        analyzer = DependencyAnalyzer()
        with pytest.raises(ValueError, match="cycle"):
            analyzer.topological_sort(graph)

    def test_execution_waves(self) -> None:
        """Test computation of execution waves for parallel scheduling."""
        # Create a graph with multiple levels
        graph = XCSGraph()
        graph.add_node(lambda x: x, "A1")
        graph.add_node(lambda x: x, "A2")
        graph.add_node(lambda x: x, "B1")
        graph.add_node(lambda x: x, "B2")
        graph.add_node(lambda x: x, "C1")
        
        # Dependencies:
        # Wave 1: A1, A2 (no dependencies)
        # Wave 2: B1 (depends on A1), B2 (depends on A2)
        # Wave 3: C1 (depends on B1 and B2)
        graph.add_edge("A1", "B1")
        graph.add_edge("A2", "B2")
        graph.add_edge("B1", "C1")
        graph.add_edge("B2", "C1")
        
        # Compute execution waves
        analyzer = DependencyAnalyzer()
        waves = analyzer.compute_execution_waves(graph)
        
        # Verify wave structure
        assert len(waves) == 3  # Three waves
        assert set(waves[0]) == {"A1", "A2"}  # First wave
        assert set(waves[1]) == {"B1", "B2"}  # Second wave
        assert waves[2] == ["C1"]  # Third wave

    def test_execution_waves_with_cycle(self) -> None:
        """Test execution wave computation fails with cycles."""
        # Create a cyclic graph
        graph = XCSGraph()
        graph.add_node(lambda x: x, "A")
        graph.add_node(lambda x: x, "B")
        
        # Set up cycle: A -> B -> A
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")
        
        # Verify cycle detection
        analyzer = DependencyAnalyzer()
        with pytest.raises(ValueError, match="cycle"):
            analyzer.compute_execution_waves(graph)