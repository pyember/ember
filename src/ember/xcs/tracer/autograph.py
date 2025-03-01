"""
Automatic Graph Building for Ember XCS.

This module provides utilities for automatically building XCS graphs from execution traces.
It analyzes trace records to identify dependencies between operators and constructs
a graph that can be used for parallel execution.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.xcs_tracing import TraceRecord

logger = logging.getLogger(__name__)


class AutoGraphBuilder:
    """Builds XCS graphs automatically from trace records.

    This class analyzes trace records to identify dependencies between operators
    and constructs a graph that can be used for parallel execution.
    """

    def __init__(self) -> None:
        """Initialize the graph builder."""
        self.dependency_map: Dict[str, Set[str]] = {}
        self.output_cache: Dict[str, Dict[str, Any]] = {}

    def build_graph(self, records: List[TraceRecord]) -> XCSGraph:
        """Build an XCS graph from a list of trace records.

        Args:
            records: List of execution trace records.

        Returns:
            An XCS graph with nodes and edges derived from the trace records.
        """
        # Create a new graph
        graph = XCSGraph()

        # Reset internal state
        self.dependency_map = {}
        self.output_cache = {}

        # First pass: add nodes and record outputs
        for idx, record in enumerate(records):
            # Create a unique node ID based on the operator name and index
            node_id = f"{record.operator_name}_{idx}"

            # Create a callable wrapper for this operation
            def create_callable(record_outputs: Any) -> Callable:
                def operation_fn(*, inputs: Dict[str, Any]) -> Any:
                    # Simply return the recorded outputs
                    # In a more sophisticated version, we could re-execute the operation
                    return record_outputs

                return operation_fn

            # Add the node to the graph
            graph.add_node(operator=create_callable(record.outputs), node_id=node_id)

            # Store outputs for dependency analysis
            self.output_cache[node_id] = record.outputs

        # Second pass: analyze dependencies and add edges
        self._analyze_dependencies(records)

        # Add the edges to the graph
        for dependent_node, dependency_nodes in self.dependency_map.items():
            for dependency in dependency_nodes:
                try:
                    graph.add_edge(from_id=dependency, to_id=dependent_node)
                except ValueError as e:
                    # Log error but continue - this might happen if nodes were pruned
                    logger.warning(f"Error adding edge: {e}")

        return graph

    def _analyze_dependencies(self, records: List[TraceRecord]) -> None:
        """Analyze dependencies between trace records.

        This method identifies which operations depend on outputs from previous operations
        by comparing input values with outputs from earlier records, with special handling
        for nested operators.

        Args:
            records: List of execution trace records.
        """
        # First pass: build a map of operator IDs to their execution order
        operator_execution_order = {}
        for idx, record in enumerate(records):
            operator_id = record.node_id
            if operator_id not in operator_execution_order:
                operator_execution_order[operator_id] = []
            operator_execution_order[operator_id].append(idx)

        # Build a map of hierarchical relationships (parent-child)
        hierarchy_map = self._build_hierarchy_map(records)

        # Second pass: analyze dependencies with hierarchy awareness
        for j, record in enumerate(records):
            dependent_node = f"{record.operator_name}_{j}"
            self.dependency_map[dependent_node] = set()

            # Check if any inputs match outputs from earlier operators
            for i in range(j):
                predecessor = f"{records[i].operator_name}_{i}"
                predecessor_outputs = self.output_cache[predecessor]

                # Skip if the nodes have a direct parent-child relationship
                # (since the child is executed "inside" the parent, not after it)
                if self._is_parent_child_relationship(
                    record.node_id, records[i].node_id, hierarchy_map
                ):
                    continue

                # Check for data dependencies
                if self._has_dependency(record.inputs, predecessor_outputs):
                    self.dependency_map[dependent_node].add(predecessor)

    def _build_hierarchy_map(self, records: List[TraceRecord]) -> Dict[str, List[str]]:
        """Build a map of hierarchical relationships between operators.

        This identifies parent-child relationships between operators based on
        execution order and operator IDs, which helps determine true dependencies.

        Args:
            records: List of execution trace records.

        Returns:
            Dictionary mapping operator IDs to lists of child operator IDs.
        """
        hierarchy_map = {}

        # Sort records by timestamp to establish execution order
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # Stack to track currently active operators
        active_operators = []

        for record in sorted_records:
            op_id = record.node_id

            # If stack is not empty, this operator might be a child of the top operator
            if active_operators:
                parent_id = active_operators[-1]
                if parent_id not in hierarchy_map:
                    hierarchy_map[parent_id] = []
                hierarchy_map[parent_id].append(op_id)

            # Add this operator to the active stack
            active_operators.append(op_id)

            # For simplicity, we pop after each operation
            # In a more sophisticated version, we would track start/end events
            active_operators.pop()

        return hierarchy_map

    def _is_parent_child_relationship(
        self, node_id1: str, node_id2: str, hierarchy_map: Dict[str, List[str]]
    ) -> bool:
        """Check if two nodes have a parent-child relationship.

        Args:
            node_id1: First node ID
            node_id2: Second node ID
            hierarchy_map: Map of hierarchical relationships

        Returns:
            True if one node is a parent of the other, False otherwise
        """
        # Check if node1 is a parent of node2
        if node_id1 in hierarchy_map and node_id2 in hierarchy_map[node_id1]:
            return True

        # Check if node2 is a parent of node1
        if node_id2 in hierarchy_map and node_id1 in hierarchy_map[node_id2]:
            return True

        return False

    def _has_dependency(self, inputs: Dict[str, Any], outputs: Any) -> bool:
        """Check if inputs depend on outputs from a previous operation.

        Args:
            inputs: Input values for an operation.
            outputs: Output values from a previous operation.

        Returns:
            True if there appears to be a dependency, False otherwise.
        """
        # If outputs is a dict-like object, check for shared keys and values
        if isinstance(outputs, dict):
            # Check for matching keys and values
            for key, value in outputs.items():
                if key in inputs and inputs[key] == value:
                    return True

            # Check if any output value appears in the inputs
            for input_key, input_value in inputs.items():
                if any(
                    output_value == input_value for _, output_value in outputs.items()
                ):
                    return True

        # If outputs is a single value that appears in inputs
        elif any(value == outputs for value in inputs.values()):
            return True

        # Special case for collection output types
        elif hasattr(outputs, "__iter__") and not isinstance(outputs, str):
            # Check if any value in the collection matches an input value
            for output_item in outputs:
                if any(input_value == output_item for input_value in inputs.values()):
                    return True

        return False
