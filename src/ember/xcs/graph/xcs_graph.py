"""
Graph representation for XCS computation.

This module provides the XCSGraph class for representing computation graphs in XCS.
It allows defining operator nodes and the edges between them, and is used by the
XCS execution engine to execute computational flows.
"""

from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple, TypeVar, cast
import dataclasses
from collections import defaultdict, deque
import uuid


@dataclasses.dataclass
class XCSNode:
    """A node in an XCS computation graph.

    Attributes:
        operator: The operator or function that performs the computation.
        node_id: Unique identifier for the node.
        inbound_edges: List of node IDs that connect to this node.
        outbound_edges: List of node IDs this node connects to.
        name: Optional human-readable name for the node.
        metadata: Optional metadata for the node (cost estimates, etc.).
    """

    operator: Callable[..., Dict[str, Any]]
    node_id: str
    inbound_edges: List[str] = dataclasses.field(default_factory=list)
    outbound_edges: List[str] = dataclasses.field(default_factory=list)
    name: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


# For backward compatibility
XCSGraphNode = XCSNode


class XCSGraph:
    """Graph representation for XCS computations.

    The XCSGraph class enables building computational graphs with operators as nodes
    and data dependencies as edges.
    """

    def __init__(self) -> None:
        """Initialize an empty XCS computation graph."""
        self.nodes: Dict[str, XCSNode] = {}
        self.metadata: Dict[str, Any] = {}

    def add_node(
        self,
        operator: Callable[..., Dict[str, Any]],
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a node to the graph.

        Args:
            operator: The operator or function to add as a node.
            node_id: Unique identifier for the node (generated if not provided).
            name: Optional human-readable name for the node.
            metadata: Optional metadata for the node (cost estimates, etc.).

        Returns:
            The ID of the added node.

        Raises:
            ValueError: If a node with the given ID already exists.
        """
        if node_id is None:
            node_id = str(uuid.uuid4())

        if node_id in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' already exists.")

        self.nodes[node_id] = XCSNode(
            operator=operator, node_id=node_id, name=name, metadata=metadata or {}
        )
        return node_id

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge between two nodes.

        Args:
            from_id: ID of the source node.
            to_id: ID of the destination node.

        Raises:
            ValueError: If either node ID does not exist.
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node '{from_id}' does not exist.")
        if to_id not in self.nodes:
            raise ValueError(f"Destination node '{to_id}' does not exist.")

        self.nodes[from_id].outbound_edges.append(to_id)
        self.nodes[to_id].inbound_edges.append(from_id)

    def topological_sort(self) -> List[str]:
        """Perform a topological sort of the graph.

        Returns:
            List of node IDs in topological order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        # Set up variables for Kahn's algorithm
        in_degree = {
            node_id: len(node.inbound_edges) for node_id, node in self.nodes.items()
        }
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        sorted_nodes = []

        # Process the queue
        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)

            for neighbor in self.nodes[current].outbound_edges:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return sorted_nodes

    def __str__(self) -> str:
        """Get a string representation of the graph.

        Returns:
            A string describing the graph.
        """
        nodes_str = [
            f"Node {node_id}: {node.name or 'unnamed'}"
            for node_id, node in self.nodes.items()
        ]
        edges_str = []
        for node_id, node in self.nodes.items():
            for edge in node.outbound_edges:
                edges_str.append(f"{node_id} -> {edge}")

        return (
            f"XCSGraph with {len(self.nodes)} nodes:\n"
            + "\n".join(nodes_str)
            + "\n\nEdges:\n"
            + "\n".join(edges_str)
        )


def merge_xcs_graphs(base: XCSGraph, additional: XCSGraph, namespace: str) -> XCSGraph:
    """Merge two XCS graphs, prefixing nodes from the additional graph with a namespace.

    Args:
        base: The base graph to merge into.
        additional: The graph to merge in, with nodes prefixed by namespace.
        namespace: Prefix to add to node IDs from the additional graph.

    Returns:
        A new merged XCSGraph instance.
    """
    merged = XCSGraph()

    # Add all nodes from the base graph
    for node_id, node in base.nodes.items():
        merged.add_node(operator=node.operator, node_id=node_id, name=node.name)

    # Add all nodes from the additional graph with namespace prefix
    node_mapping = {}  # Maps original IDs to namespaced IDs
    for node_id, node in additional.nodes.items():
        namespaced_id = f"{namespace}_{node_id}"
        # If the node ID exists in the base graph, ensure uniqueness
        if namespaced_id in merged.nodes:
            namespaced_id = f"{namespace}_{node_id}_{uuid.uuid4().hex[:8]}"

        merged.add_node(operator=node.operator, node_id=namespaced_id, name=node.name)
        node_mapping[node_id] = namespaced_id

    # Add edges from the base graph
    for node_id, node in base.nodes.items():
        for edge_to in node.outbound_edges:
            merged.add_edge(from_id=node_id, to_id=edge_to)

    # Add edges from the additional graph, using namespaced IDs
    for node_id, node in additional.nodes.items():
        from_id = node_mapping[node_id]
        for edge_to in node.outbound_edges:
            to_id = node_mapping.get(edge_to, edge_to)  # Use mapped ID if it exists
            merged.add_edge(from_id=from_id, to_id=to_id)

    return merged
