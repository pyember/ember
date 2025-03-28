"""Unit tests for the autograph module.

Tests the dependency analysis and graph construction functionality:
- AutoGraphBuilder: Graph construction from execution traces
- DependencyAnalyzer: Dependency detection between operator invocations
- DataReference: Data object reference tracking
- JITCache: Caching system with weak references and state dependency invalidation
"""

import unittest
import weakref

from ember.xcs.tracer.autograph import (
    AutoGraphBuilder,
    DataReference,
    DependencyAnalyzer,
)
from ember.xcs.tracer.tracer_decorator import JITCache, StateDependency
from ember.xcs.tracer.xcs_tracing import TraceRecord


class TestAutoGraphBuilder(unittest.TestCase):
    """Tests AutoGraphBuilder's ability to construct graphs from traces.

    Validates:
    - Simple sequential graph construction
    - Branching dependency detection in diamond patterns
    - Low-level dependency analysis mechanisms
    """

    def test_build_simple_graph(self) -> None:
        """Verifies construction of a sequential linear graph from trace records.

        Creates a chain of three operators with sequential data dependencies
        and confirms the resulting graph preserves execution order.
        """
        # Create a series of trace records that form a linear chain
        records = [
            TraceRecord(
                operator_name="Op1",
                node_id="1",
                inputs={"query": "test"},
                outputs={"result": "output1"},
            ),
            TraceRecord(
                operator_name="Op2",
                node_id="2",
                inputs={"result": "output1"},
                outputs={"intermediate": "output2"},
            ),
            TraceRecord(
                operator_name="Op3",
                node_id="3",
                inputs={"intermediate": "output2"},
                outputs={"final": "output3"},
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
        """Verifies construction of a diamond-shaped graph with branches.

        Creates a more complex graph with multiple execution paths and validates
        that parallel branches are correctly captured in the resulting graph.
        """
        # Create trace records that form a diamond pattern:
        # Op1 -> Op2a -> Op3
        #   \-> Op2b -/
        records = [
            TraceRecord(
                operator_name="Op1",
                node_id="1",
                inputs={"query": "test"},
                outputs={"result": "output1"},
            ),
            TraceRecord(
                operator_name="Op2a",
                node_id="2a",
                inputs={"result": "output1"},
                outputs={"branch_a": "output2a"},
            ),
            TraceRecord(
                operator_name="Op2b",
                node_id="2b",
                inputs={"result": "output1"},
                outputs={"branch_b": "output2b"},
            ),
            TraceRecord(
                operator_name="Op3",
                node_id="3",
                inputs={"branch_a": "output2a", "branch_b": "output2b"},
                outputs={"final": "output3"},
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

    def test_dependency_detection(self) -> None:
        """Tests the core dependency detection mechanisms.

        Validates both data signature matching and object identity tracking
        to ensure dependencies are correctly identified between operations.
        """
        analyzer = DependencyAnalyzer()

        # Test data signature matching
        ref1 = DataReference(
            obj_id=None, path="test", signature="string:output_from_previous"
        )
        ref2 = DataReference(
            obj_id=None, path="test", signature="string:output_from_previous"
        )

        # Same signature should match
        self.assertEqual(ref1.signature, ref2.signature)

        # Test object identity matching
        obj = {"key": "value"}
        ref3 = DataReference(obj_id=id(obj), path="test", signature="obj:signature")
        ref4 = DataReference(obj_id=id(obj), path="test2", signature="different")

        # Same object ID should be detected
        self.assertEqual(ref3.obj_id, ref4.obj_id)

        # Test data flow analysis
        records = [
            TraceRecord(
                operator_name="Op1",
                node_id="1",
                inputs={"query": "test"},
                outputs={"result": "output1"},
            ),
            TraceRecord(
                operator_name="Op2",
                node_id="2",
                inputs={"data": "output1"},
                outputs={"final": "output2"},
            ),
        ]

        # Analyze dependencies
        dep_nodes = analyzer.analyze(records)

        # Op2 should depend on Op1
        self.assertIn("1", dep_nodes["2"].dependencies)


class TestJITRefactoring(unittest.TestCase):
    """Tests JIT compilation components and operator execution.

    Validates:
    - JITCache weak reference cleanup for memory efficiency
    - State signature-based cache invalidation
    - Operator callable execution with real operators
    - Fallback behavior when operators are no longer available
    """

    def test_jit_cache_with_weak_references(self):
        """Verifies JITCache's memory management with weak references.

        Tests that cache entries are automatically removed when their keys
        are garbage collected, preventing memory leaks.
        """
        cache = JITCache()

        # Create a test object
        class TestObject:
            pass

        obj = TestObject()

        # Add to cache
        cache.set(obj, "test_value")

        # Verify it's in the cache
        self.assertEqual(cache.get(obj), "test_value")
        self.assertEqual(len(cache), 1)

        # Remove all references to obj
        obj_ref = weakref.ref(obj)
        del obj

        # Force garbage collection
        import gc

        gc.collect()

        # Verify the object was garbage collected
        self.assertIsNone(obj_ref())

        # Verify the cache is now empty
        self.assertEqual(len(cache), 0)

    def test_state_dependency_cache_invalidation(self):
        """Verifies cache invalidation based on state signatures.

        Tests the StateDependency protocol implementation with JITCache,
        ensuring cached values are invalidated when state changes.
        """
        cache = JITCache()

        # Create a test object with state
        class TestObjectWithState(StateDependency):
            def __init__(self):
                self.state_version = 1

            def get_state_signature(self):
                return f"state_v{self.state_version}"

            def get_state_dependencies(self):
                return set()  # Empty set as per the StateDependency protocol

        obj = TestObjectWithState()

        # Add to cache with initial state
        cache.set(obj, "value1", obj.get_state_signature())

        # Verify we can get the value
        self.assertEqual(cache.get_with_state(obj, obj.get_state_signature()), "value1")

        # Change state
        obj.state_version = 2

        # Verify the value is no longer accessible with new state
        self.assertIsNone(cache.get_with_state(obj, obj.get_state_signature()))

        # Set a new value with the new state
        cache.set(obj, "value2", obj.get_state_signature())

        # Verify we can get the new value
        self.assertEqual(cache.get_with_state(obj, obj.get_state_signature()), "value2")

    def test_operator_callable_execution(self):
        """Tests the operator callable mechanism with live execution.

        Verifies that operator callables created from trace records properly:
        1. Execute the actual operator with new inputs when available
        2. Fall back to traced outputs when operators are unavailable
        """

        # Create a simple operator and trace record
        class SimpleOperator:
            def __init__(self):
                self.call_count = 0

            def __call__(self, *, inputs):
                self.call_count += 1
                return {"result": inputs["value"] * 2}

        operator = SimpleOperator()

        # Create a trace record with the operator
        record = TraceRecord(
            operator_name="SimpleOperator",
            node_id="1",
            inputs={"value": 5},
            outputs={"result": 10},
            operator=operator,
        )

        # Create the callable using AutoGraphBuilder's method
        builder = AutoGraphBuilder()
        op_callable = builder._create_operator_callable(trace_record=record)

        # Execute the callable with a new input
        result = op_callable(inputs={"value": 7})

        # Verify the operator was actually called and result is as expected
        self.assertEqual(operator.call_count, 1)
        self.assertEqual(result, {"result": 14})

        # Test fallback behavior when operator is gone
        record.operator = None
        op_callable = builder._create_operator_callable(trace_record=record)

        # This should return the original outputs from the trace record
        result = op_callable(inputs={"value": 20})
        self.assertEqual(result, {"result": 10})


if __name__ == "__main__":
    unittest.main()
