"""Unit tests for the autograph module.

This module contains tests for the AutoGraphBuilder class, which is responsible
for automatically building XCS graphs from execution traces.
"""

import unittest
from typing import Any, Dict, List

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.autograph import AutoGraphBuilder
from ember.xcs.tracer.xcs_tracing import TraceRecord


class TestAutoGraphBuilder(unittest.TestCase):
    """Tests for the AutoGraphBuilder class."""
    
    def test_build_simple_graph(self) -> None:
        """Test building a simple graph with sequential dependencies."""
        # Create a series of trace records that form a linear chain
        records = [
            TraceRecord(
                operator_name="Op1",
                node_id="1",
                inputs={"query": "test"},
                outputs={"result": "output1"},
                timestamp=1.0,
            ),
            TraceRecord(
                operator_name="Op2",
                node_id="2",
                inputs={"result": "output1"},
                outputs={"intermediate": "output2"},
                timestamp=2.0,
            ),
            TraceRecord(
                operator_name="Op3",
                node_id="3",
                inputs={"intermediate": "output2"},
                outputs={"final": "output3"},
                timestamp=3.0,
            ),
        ]
        
        # Build the graph
        builder = AutoGraphBuilder()
        graph = builder.build_graph(records)
        
        # Verify the graph structure
        self.assertEqual(len(graph.nodes), 3)
        self.assertIn("Op1_0", graph.nodes)
        self.assertIn("Op2_1", graph.nodes)
        self.assertIn("Op3_2", graph.nodes)
        
        # Check dependencies - Op2 should depend on Op1, Op3 should depend on Op2
        self.assertIn("Op1_0", graph.nodes["Op2_1"].inbound_edges)
        self.assertIn("Op2_1", graph.nodes["Op3_2"].inbound_edges)
    
    def test_build_graph_with_branches(self) -> None:
        """Test building a graph with branching dependencies."""
        # Create trace records that form a diamond pattern:
        # Op1 -> Op2a -> Op3
        #   \-> Op2b -/
        records = [
            TraceRecord(
                operator_name="Op1",
                node_id="1",
                inputs={"query": "test"},
                outputs={"result": "output1"},
                timestamp=1.0,
            ),
            TraceRecord(
                operator_name="Op2a",
                node_id="2a",
                inputs={"result": "output1"},
                outputs={"branch_a": "output2a"},
                timestamp=2.0,
            ),
            TraceRecord(
                operator_name="Op2b",
                node_id="2b",
                inputs={"result": "output1"},
                outputs={"branch_b": "output2b"},
                timestamp=2.1,
            ),
            TraceRecord(
                operator_name="Op3",
                node_id="3",
                inputs={"branch_a": "output2a", "branch_b": "output2b"},
                outputs={"final": "output3"},
                timestamp=3.0,
            ),
        ]
        
        # Build the graph
        builder = AutoGraphBuilder()
        graph = builder.build_graph(records)
        
        # Verify the graph structure
        self.assertEqual(len(graph.nodes), 4)
        
        # Check dependencies
        self.assertIn("Op1_0", graph.nodes["Op2a_1"].inbound_edges)
        self.assertIn("Op1_0", graph.nodes["Op2b_2"].inbound_edges)
        self.assertIn("Op2a_1", graph.nodes["Op3_3"].inbound_edges)
        self.assertIn("Op2b_2", graph.nodes["Op3_3"].inbound_edges)
        
    def test_build_graph_with_nested_operators(self) -> None:
        """Test building a graph with nested operator calls.
        
        This scenario simulates a parent operator that calls child operators,
        where the child operators should not be directly dependent on the parent.
        """
        # Create trace records that represent a nested execution pattern:
        # ParentOp calls ChildOp1 and ChildOp2 internally, which shouldn't be treated as dependencies
        records = [
            TraceRecord(
                operator_name="ParentOp",
                node_id="parent1",
                inputs={"query": "test"},
                outputs={"result": "parent_output"},
                timestamp=1.0,
            ),
            TraceRecord(
                operator_name="ChildOp1",
                node_id="child1",
                inputs={"query": "test"},
                outputs={"child_result": "child1_output"},
                timestamp=1.1,  # Executed during ParentOp
            ),
            TraceRecord(
                operator_name="ChildOp2",
                node_id="child2",
                inputs={"child_input": "child1_output"},
                outputs={"final": "child2_output"},
                timestamp=1.2,  # Executed during ParentOp
            ),
            TraceRecord(
                operator_name="NextOp",
                node_id="next1",
                inputs={"result": "parent_output"},
                outputs={"next_result": "next_output"},
                timestamp=2.0,  # Executed after ParentOp
            ),
        ]
        
        # Build the graph
        builder = AutoGraphBuilder()
        
        # Create a manual hierarchy map to simulate nested execution
        hierarchy_map = {"parent1": ["child1", "child2"]}
        builder._build_hierarchy_map = lambda records: hierarchy_map
        
        graph = builder.build_graph(records)
        
        # Verify the graph structure
        self.assertEqual(len(graph.nodes), 4)
        
        # Check dependencies - NextOp should depend on ParentOp, not on the child ops
        self.assertIn("ParentOp_0", graph.nodes["NextOp_3"].inbound_edges)
        
        # Verify that NextOp does not directly depend on children
        self.assertNotIn("ChildOp1_1", graph.nodes["NextOp_3"].inbound_edges)
        self.assertNotIn("ChildOp2_2", graph.nodes["NextOp_3"].inbound_edges)
        
    def test_complex_execution_flow(self) -> None:
        """Test building a graph with a complex execution flow.
        
        This test simulates a complex execution with multiple paths and
        nested operators, to verify that the dependency analysis correctly
        handles complex flows.
        """
        # Simulate a more complex execution pattern with multiple nested operators
        # and branches:
        #
        # Main -> BranchA -> SubA1 -> MergePoint
        #     \-> BranchB -> SubB1 -/
        #                \-> SubB2
        
        records = [
            # Main operator
            TraceRecord(
                operator_name="Main",
                node_id="main",
                inputs={"input": "start"},
                outputs={"main_out": "main_result"},
                timestamp=1.0,
            ),
            # Branch A
            TraceRecord(
                operator_name="BranchA",
                node_id="branchA",
                inputs={"main_out": "main_result"},
                outputs={"branch_a_out": "branch_a_result"},
                timestamp=2.0,
            ),
            # Sub-operator A1 (runs inside Branch A)
            TraceRecord(
                operator_name="SubA1",
                node_id="subA1",
                inputs={"branch_a_data": "some_data"},
                outputs={"subA1_out": "subA1_result"},
                timestamp=2.1,
            ),
            # Branch B
            TraceRecord(
                operator_name="BranchB",
                node_id="branchB",
                inputs={"main_out": "main_result"},
                outputs={"branch_b_out": "branch_b_result"},
                timestamp=3.0,
            ),
            # Sub-operator B1 (runs inside Branch B)
            TraceRecord(
                operator_name="SubB1",
                node_id="subB1",
                inputs={"branch_b_data": "some_b_data"},
                outputs={"subB1_out": "subB1_result"},
                timestamp=3.1,
            ),
            # Sub-operator B2 (runs inside Branch B)
            TraceRecord(
                operator_name="SubB2",
                node_id="subB2",
                inputs={"subB1_out": "subB1_result"},
                outputs={"subB2_out": "subB2_result"},
                timestamp=3.2,
            ),
            # Merge point that depends on both branches
            TraceRecord(
                operator_name="MergePoint",
                node_id="merge",
                inputs={
                    "branch_a_out": "branch_a_result", 
                    "subB1_out": "subB1_result"
                },
                outputs={"final": "merged_result"},
                timestamp=4.0,
            ),
        ]
        
        # Create a manual hierarchy map to simulate nested execution
        hierarchy_map = {
            "branchA": ["subA1"],
            "branchB": ["subB1", "subB2"],
        }
        
        # Build the graph
        builder = AutoGraphBuilder()
        builder._build_hierarchy_map = lambda records: hierarchy_map
        graph = builder.build_graph(records)
        
        # Verify correct dependencies
        self.assertIn("Main_0", graph.nodes["BranchA_1"].inbound_edges)
        self.assertIn("Main_0", graph.nodes["BranchB_3"].inbound_edges)
        self.assertIn("BranchA_1", graph.nodes["MergePoint_6"].inbound_edges)
        
        # MergePoint should depend on SubB1, but not on SubB2 since it doesn't use its output
        self.assertIn("SubB1_4", graph.nodes["MergePoint_6"].inbound_edges)
    
    def test_dependency_detection(self) -> None:
        """Test the dependency detection logic."""
        builder = AutoGraphBuilder()
        
        # Test dictionary value dependency
        inputs = {"x": 1, "y": "output_from_previous"}
        outputs = {"result": "output_from_previous"}
        self.assertTrue(builder._has_dependency(inputs, outputs))
        
        # Test no dependency
        inputs = {"x": 1, "y": "no_match"}
        outputs = {"result": "output_from_previous"}
        self.assertFalse(builder._has_dependency(inputs, outputs))
        
        # Test direct value dependency
        inputs = {"x": 1, "y": "output_value"}
        outputs = "output_value"
        self.assertTrue(builder._has_dependency(inputs, outputs))


if __name__ == "__main__":
    unittest.main()