"""
Automatic Graph Building for Ember XCS.

This module provides utilities for automatically building XCS graphs from execution
traces. It analyzes trace records to identify dependencies between operators and
constructs a graph that can be used for parallel execution.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Set

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.xcs_tracing import TraceRecord

logger = logging.getLogger(__name__)


class AutoGraphBuilder:
    """Constructs XCS graphs automatically from execution trace records.

    This class parses execution trace records to identify operator dependencies and
    builds an XCSGraph instance ready for parallel execution.
    """

    def __init__(self) -> None:
        """Initializes an AutoGraphBuilder instance."""
        self.dependency_map: Dict[str, Set[str]] = {}
        self.output_cache: Dict[str, Dict[str, Any]] = {}

    def build_graph(self, records: List[TraceRecord] = None, **kwargs) -> XCSGraph:
        """Builds an XCS graph from the provided trace records.

        This method resets the internal state, adds nodes to the graph while caching
        their outputs, analyzes dependency relationships among the records, and finally
        connects nodes by adding the appropriate edges.

        Args:
            records: A list of TraceRecord instances representing the execution trace.
                Can be positional or provided via the 'records' keyword argument.

        Returns:
            XCSGraph: An instance of XCSGraph populated with nodes and interconnecting edges.
        """
        # Support both positional and keyword args to handle different calling conventions
        if records is None and 'records' in kwargs:
            records = kwargs['records']
            
        graph: XCSGraph = XCSGraph()

        # Reset internal state.
        self.dependency_map.clear()
        self.output_cache.clear()

        # First pass: Add nodes to the graph and cache their outputs.
        for i, record in enumerate(records):
            node_id: str = record.node_id
            # Generate predictable node IDs for tests that match the expected format
            graph_node_id = f"{record.operator_name}_{i}"
            
            operator_callable: Callable[[Dict[str, Any]], Any] = self._create_operator_callable(
                record_outputs=record.outputs
            )
            graph.add_node(operator=operator_callable, node_id=graph_node_id)
            self.output_cache[node_id] = record.outputs
            # Map original node_id to graph_node_id for dependency building
            record.graph_node_id = graph_node_id

        # Second pass: Analyze dependencies and add connecting edges.
        self._analyze_dependencies(records=records)
        for dependent_node, dependency_nodes in self.dependency_map.items():
            # Find the corresponding graph node for the dependent node
            dependent_graph_node = next((r.graph_node_id for r in records if r.node_id == dependent_node), None)
            
            for dependency in dependency_nodes:
                try:
                    # Find the corresponding graph node for the dependency
                    dependency_graph_node = next((r.graph_node_id for r in records if r.node_id == dependency), None)
                    if dependent_graph_node and dependency_graph_node:
                        graph.add_edge(from_id=dependency_graph_node, to_id=dependent_graph_node)
                except ValueError as error:
                    logger.warning(
                        "Error adding edge from '%s' to '%s': %s",
                        dependency,
                        dependent_node,
                        error,
                    )

        return graph

    @staticmethod
    def _create_operator_callable(*, record_outputs: Any) -> Callable[[Dict[str, Any]], Any]:
        """Creates a callable operation that returns the provided outputs.

        Args:
            record_outputs: The output value recorded for an operator.

        Returns:
            Callable[[Dict[str, Any]], Any]: A callable that returns 'record_outputs' when
            invoked with the keyword argument 'inputs'.
        """
        def operation_fn(*, inputs: Dict[str, Any]) -> Any:
            return record_outputs

        return operation_fn

    def _analyze_dependencies(self, *, records: List[TraceRecord]) -> None:
        """Analyzes and records dependencies between execution trace records.

        This method examines each record's inputs to determine if they depend on the outputs
        of any preceding operation while respecting potential parent-child (hierarchical)
        relationships.

        Args:
            records: A list of TraceRecord instances to analyze.
        """
        hierarchy_map: Dict[str, List[str]] = self._build_hierarchy_map(records=records)

        for index, record in enumerate(records):
            dependent_node: str = record.node_id
            self.dependency_map[dependent_node] = set()

            for prev_index in range(index):
                predecessor_node: str = records[prev_index].node_id
                predecessor_outputs: Any = self.output_cache.get(predecessor_node)
                # Skip dependency if a direct parent-child relationship exists.
                if self._is_parent_child_relationship(
                    node_id1=record.node_id,
                    node_id2=records[prev_index].node_id,
                    hierarchy_map=hierarchy_map,
                ):
                    continue

                if self._has_dependency(inputs=record.inputs, outputs=predecessor_outputs):
                    self.dependency_map[dependent_node].add(predecessor_node)

    def _build_hierarchy_map(self, *, records: List[TraceRecord]) -> Dict[str, List[str]]:
        """Builds a mapping of parent-child relationships between operators.

        Hierarchical relationships are inferred from the execution order based on record
        timestamps.

        Args:
            records: A list of TraceRecord instances.

        Returns:
            Dict[str, List[str]]: A dictionary mapping a parent operator's node ID to the list
            of its child node IDs.
        """
        hierarchy_map: Dict[str, List[str]] = {}
        sorted_records: List[TraceRecord] = sorted(records, key=lambda r: r.timestamp)
        active_operators: List[str] = []

        for record in sorted_records:
            op_id: str = record.node_id
            if active_operators:
                parent_id: str = active_operators[-1]
                hierarchy_map.setdefault(parent_id, []).append(op_id)

            active_operators.append(op_id)
            # For simplicity, assume each operation completes immediately.
            active_operators.pop()

        return hierarchy_map

    def _is_parent_child_relationship(
        self, *, node_id1: str, node_id2: str, hierarchy_map: Dict[str, List[str]]
    ) -> bool:
        """Determines if two nodes share a parent-child hierarchical relationship.

        Args:
            node_id1: The identifier of the first node.
            node_id2: The identifier of the second node.
            hierarchy_map: A mapping from parent node IDs to lists of child node IDs.

        Returns:
            bool: True if one node is the parent of the other; otherwise, False.
        """
        if node_id1 in hierarchy_map and node_id2 in hierarchy_map[node_id1]:
            return True

        if node_id2 in hierarchy_map and node_id1 in hierarchy_map[node_id2]:
            return True

        return False

    def _has_dependency(self, *, inputs: Dict[str, Any], outputs: Any) -> bool:
        """Checks whether the provided inputs depend on outputs from a previous operation.

        Dependencies are detected by comparing input values against output values.

        Args:
            inputs: A dictionary of input values for an operator.
            outputs: The output(s) from a preceding operation; its type is arbitrary.

        Returns:
            bool: True if a dependency is detected; otherwise, False.
        """
        if isinstance(outputs, dict):
            # Check for matching keys with identical values.
            for key, value in outputs.items():
                if key in inputs and inputs[key] == value:
                    return True
            # Additionally, check if any output value appears among the inputs.
            for input_value in inputs.values():
                if any(output_value == input_value for output_value in outputs.values()):
                    return True
        elif any(value == outputs for value in inputs.values()):
            return True
        elif hasattr(outputs, "__iter__") and not isinstance(outputs, str):
            for output_item in outputs:
                if any(input_value == output_item for input_value in inputs.values()):
                    return True

        return False
