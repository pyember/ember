"""Computation graph for XCS execution.

Defines a directed acyclic graph structure for representing and executing
computational flows. Operators form nodes in the graph, with edges representing
data dependencies between operations.

Example:
    ```python
    graph = XCSGraph()
    
    # Add computation nodes
    input_node = graph.add_node(preprocess_fn, name="preprocess")
    compute_node = graph.add_node(compute_fn, name="compute")
    output_node = graph.add_node(postprocess_fn, name="postprocess")
    
    # Define data flow
    graph.add_edge(input_node, compute_node)
    graph.add_edge(compute_node, output_node)
    
    # Execute the computation with an execution engine
    from ember.xcs.engine import execute
    results = execute(graph, inputs={"data": input_data})
    ```
"""

import dataclasses
import uuid
from collections import deque
from typing import Any, Callable, Dict, List, Optional


@dataclasses.dataclass
class XCSEdge:
    """Edge connecting two nodes with field-level mapping information.

    Represents a data dependency between nodes with precise information about
    which output fields connect to which input fields.

    Attributes:
        from_node: Source node ID producing data
        to_node: Destination node ID consuming data
        field_mappings: Maps output fields to input fields for precise data flow
    """

    from_node: str
    to_node: str
    field_mappings: Dict[str, str] = dataclasses.field(default_factory=dict)

    def add_field_mapping(self, output_field: str, input_field: str) -> None:
        """Add mapping from output field to input field.

        Args:
            output_field: Field name in the source node's output
            input_field: Field name in the destination node's input
        """
        self.field_mappings[output_field] = input_field


@dataclasses.dataclass
class XCSNode:
    """Single computation node in an execution graph.

    Represents one operation in a computational flow with its connections
    to other nodes. Each node contains an executable operator and maintains
    its position in the graph through edge lists.

    Attributes:
        operator: Callable function or operator executing this node's computation
        node_id: Unique identifier for addressing this node in the graph
        inbound_edges: Node IDs that provide inputs to this node
        outbound_edges: Node IDs that consume output from this node
        name: Human-readable label for debugging and visualization
        metadata: Additional node properties (e.g., cost estimates, device placement)
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
    """Directed graph for computational workflows.

    Provides a structure for defining complex computational flows as directed
    graphs. Supports operations needed for graph analysis, transformation, and
    execution by the XCS execution engine.
    """

    def __init__(self) -> None:
        """Creates an empty computation graph."""
        self.nodes: Dict[str, XCSNode] = {}
        self.edges: Dict[str, XCSEdge] = {}  # Edge registry for field mappings
        self.metadata: Dict[str, Any] = {}

    def add_node(
        self,
        operator: Callable[..., Dict[str, Any]],
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_key: Optional[str] = None,  # Backward compatibility
        input_mapping: Optional[Dict[str, str]] = None,  # Backward compatibility
    ) -> str:
        """Adds a computation node to the graph.

        Args:
            operator: Function or operator to execute at this node
            node_id: Unique identifier (auto-generated if None)
            name: Human-readable label for the node
            metadata: Additional properties for analysis and optimization
            node_key: (Backward compatibility) Alternative name, takes precedence over name
            input_mapping: (Backward compatibility) Field mappings for node inputs
            name: Human-readable label for the node
            metadata: Additional properties for analysis and optimization

        Returns:
            Generated or provided node ID

        Raises:
            ValueError: If node_id already exists in the graph
        """
        # Handle backward compatibility parameters
        if node_key is not None:
            name = node_key  # node_key takes precedence over name for backward compatibility

        if node_id is None:
            node_id = str(uuid.uuid4())

        if node_id in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' already exists.")

        # Create actual node with combined metadata
        node_metadata = metadata or {}
        if input_mapping:
            node_metadata["input_mapping"] = input_mapping
            
        self.nodes[node_id] = XCSNode(
            operator=operator, node_id=node_id, name=name, metadata=node_metadata
        )
        
        return node_id

    def add_edge(
        self, from_id: str, to_id: str, field_mappings: Optional[Dict[str, str]] = None
    ) -> XCSEdge:
        """Creates a directed data dependency between nodes.

        Establishes that the output of one node flows into another,
        forming a directed edge in the computation graph.

        Args:
            from_id: Source node producing output data
            to_id: Destination node consuming the data
            field_mappings: Optional mapping from output fields to input fields

        Returns:
            The created edge object

        Raises:
            ValueError: If either node doesn't exist in the graph
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node '{from_id}' does not exist.")
        if to_id not in self.nodes:
            raise ValueError(f"Destination node '{to_id}' does not exist.")

        # Create or retrieve the edge
        edge_key = f"{from_id}_{to_id}"
        if edge_key not in self.edges:
            edge = XCSEdge(from_node=from_id, to_node=to_id)
            self.edges[edge_key] = edge
        else:
            edge = self.edges[edge_key]

        # Add field mappings if provided
        if field_mappings:
            for output_field, input_field in field_mappings.items():
                edge.add_field_mapping(output_field, input_field)

        # Maintain backward compatibility with node edge lists
        self.nodes[from_id].outbound_edges.append(to_id)
        self.nodes[to_id].inbound_edges.append(from_id)

        return edge

    def topological_sort(self) -> List[str]:
        """Orders nodes so dependencies come before dependents.

        Produces an execution ordering where each node appears after
        all nodes it depends on, ensuring valid sequential execution.

        Returns:
            List of node IDs in dependency-respecting order

        Raises:
            ValueError: If graph contains cycles (not a DAG)
        """
        # Track remaining dependencies for each node
        in_degree = {
            node_id: len(node.inbound_edges) for node_id, node in self.nodes.items()
        }
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        sorted_nodes = []

        # Process nodes in topological order
        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)

            for neighbor in self.nodes[current].outbound_edges:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Verify complete ordering (no cycles)
        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return sorted_nodes

    def prepare_node_inputs(
        self, node_id: str, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepares inputs for a node based on edge field mappings.

        Args:
            node_id: The node to prepare inputs for
            results: Dictionary mapping node IDs to their output results

        Returns:
            Dictionary of inputs prepared for the node's execution
        """
        # Build inputs dictionary from upstream results
        inputs = {}
        incoming_node_ids = self.nodes[node_id].inbound_edges

        for from_id in incoming_node_ids:
            edge_key = f"{from_id}_{node_id}"
            if edge_key not in self.edges or from_id not in results:
                continue

            edge = self.edges[edge_key]
            source_results = results[from_id]

            # Map fields according to edge mappings
            if edge.field_mappings:
                for output_field, input_field in edge.field_mappings.items():
                    if output_field in source_results:
                        inputs[input_field] = source_results[output_field]
            else:
                # Default behavior: merge all results
                inputs.update(source_results)
        
        return inputs

    def __str__(self) -> str:
        """Creates a human-readable graph representation.

        Generates a structured text description showing nodes and
        their connections, useful for debugging and visualization.

        Returns:
            Multi-line string describing the graph structure
        """
        nodes_str = [
            f"Node {node_id}: {node.name or 'unnamed'}"
            for node_id, node in self.nodes.items()
        ]
        edges_str = []
        for edge_key, edge in self.edges.items():
            field_str = ""
            if edge.field_mappings:
                mappings = ", ".join(
                    f"{k}->{v}" for k, v in edge.field_mappings.items()
                )
                field_str = f" ({mappings})"
            edges_str.append(f"{edge.from_node} -> {edge.to_node}{field_str}")

        return (
            f"XCSGraph with {len(self.nodes)} nodes, {len(self.edges)} edges:\n"
            + "\n".join(nodes_str)
            + "\n\nEdges:\n"
            + "\n".join(edges_str)
        )


def merge_xcs_graphs(base: XCSGraph, additional: XCSGraph, namespace: str) -> XCSGraph:
    """Combines two computation graphs with namespace isolation.

    Creates a new graph containing all nodes from both input graphs,
    with nodes from the additional graph prefixed to avoid collisions.
    Preserves all edge connections and field mappings, adjusting IDs as needed.

    Args:
        base: Primary graph to merge into
        additional: Secondary graph to incorporate with namespace prefixing
        namespace: Prefix for additional graph's node IDs for isolation

    Returns:
        New graph containing nodes and edges from both inputs

    Example:
        ```python
        # Merge specialized processing graph into main workflow
        main_graph = XCSGraph()  # Main computation pipeline
        process_graph = XCSGraph()  # Specialized processing subgraph

        # Combine while isolating process_graph nodes
        merged = merge_xcs_graphs(main_graph, process_graph, "process")
        ```
    """
    merged = XCSGraph()

    # Copy base graph nodes with original IDs
    for node_id, node in base.nodes.items():
        merged.add_node(
            operator=node.operator,
            node_id=node_id,
            name=node.name,
            metadata=node.metadata.copy(),
        )

    # Copy additional graph nodes with namespaced IDs to prevent collisions
    node_mapping = {}  # Maps original IDs to namespaced IDs
    for node_id, node in additional.nodes.items():
        namespaced_id = f"{namespace}_{node_id}"
        # Ensure uniqueness with random suffix if needed
        if namespaced_id in merged.nodes:
            namespaced_id = f"{namespace}_{node_id}_{uuid.uuid4().hex[:8]}"

        merged.add_node(
            operator=node.operator,
            node_id=namespaced_id,
            name=node.name,
            metadata=node.metadata.copy(),
        )
        node_mapping[node_id] = namespaced_id

    # Recreate edge connections from base graph (unchanged)
    for edge_key, edge in base.edges.items():
        # Copy edge with its field mappings
        merged.add_edge(
            from_id=edge.from_node,
            to_id=edge.to_node,
            field_mappings=edge.field_mappings.copy() if edge.field_mappings else None,
        )

    # Recreate edge connections from additional graph (with ID translation)
    for edge_key, edge in additional.edges.items():
        # Translate source and destination IDs
        from_id = node_mapping.get(edge.from_node, edge.from_node)
        to_id = node_mapping.get(edge.to_node, edge.to_node)

        # Copy edge with its field mappings
        merged.add_edge(
            from_id=from_id,
            to_id=to_id,
            field_mappings=edge.field_mappings.copy() if edge.field_mappings else None,
        )

    # Merge metadata (without overwriting)
    for key, value in base.metadata.items():
        merged.metadata[key] = value

    # Add additional metadata with namespace prefix to avoid collisions
    for key, value in additional.metadata.items():
        merged.metadata[f"{namespace}_{key}"] = value

    return merged
