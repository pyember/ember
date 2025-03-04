"""
XCS Graph: Core Intermediate Representation

This module provides the canonical intermediate representation (IR) for the XCS system
in the form of a directed acyclic graph (DAG). The graph serves as the foundation for
operator composition, dependency tracking, and execution scheduling.

Key features:
- Strongly-typed nodes and edges with robust validation
- Rich metadata support for scheduling hints and optimization
- Topological operations for dependency resolution and execution ordering
- Support for graph composition and transformation

The XCSGraph is designed to be both human-readable for debugging and
machine-optimizable for high-performance execution. It implements the
Single Responsibility Principle by focusing exclusively on graph structure
and properties, delegating execution concerns to the engine module.
"""

import uuid
import logging
from typing import (
    Dict, 
    List, 
    Optional, 
    TypeVar, 
    Generic, 
    Callable, 
    Mapping, 
    cast,
    Set,
    Iterator
)
from typing_extensions import TypedDict, NotRequired

from ember.core.types.xcs_types import (
    NodeInputT,
    NodeOutputT,
    XCSNodeAttributes,
    NodeMetadata
)

logger: logging.Logger = logging.getLogger(__name__)

# Type variables for node input/output types
I = TypeVar('I', bound=Mapping[str, object])
O = TypeVar('O', bound=Mapping[str, object])


class CapturedInputs(TypedDict, total=False):
    """TypedDict for captured inputs during execution."""
    
    prompts: Optional[List[str]]
    parameters: NotRequired[Mapping[str, object]]
    timestamp: NotRequired[float]
    call_id: NotRequired[str]
    trace_id: NotRequired[str]


class XCSNode(Generic[I, O]):
    """
    Represents a single computation unit in the XCS graph IR.

    An XCSNode encapsulates a callable operator with its connectivity information and metadata.
    Each node functions as a self-contained unit with clearly defined inputs, outputs, and
    dependencies within the computation graph. The node's attributes provide critical 
    scheduling hints, optimization directives, and execution configuration.

    The design follows the Interface Segregation and Single Responsibility principles:
    - Node structure is separated from execution logic
    - Edge connections provide clear dependency contracts
    - Attributes offer a standardized extension mechanism for scheduling hints
    - Captured inputs/outputs facilitate debugging and introspection

    Attributes:
        node_id: Unique string identifier for the node within the graph.
        operator: Callable that implements the node's computation logic.
        inbound_edges: List of node IDs whose outputs are consumed by this node.
        outbound_edges: List of node IDs that consume this node's outputs.
        attributes: Extensible metadata dictionary for scheduling hints and optimization directives.
        captured_inputs: Debug and introspection data for the node's inputs during execution.
        captured_outputs: The outputs produced by the operator after execution, if available.
    """

    def __init__(self, operator: Callable[[I], O], node_id: Optional[str] = None) -> None:
        """Initializes an XCSNode instance.

        Args:
            operator: The operator callable for this node.
            node_id: An optional node identifier; if not provided, a new unique ID is generated.
        """
        if node_id is None:
            node_id = str(uuid.uuid4())[:8]
        self.node_id: str = node_id
        self.operator: Callable[[I], O] = operator
        self.inbound_edges: List[str] = []
        self.outbound_edges: List[str] = []
        self.attributes: XCSNodeAttributes = {}
        self.captured_inputs: CapturedInputs = {"prompts": None}
        self.captured_outputs: Optional[O] = None

    def add_inbound_edge(self, *, from_id: str) -> None:
        """Adds an inbound edge from the specified node.

        Args:
            from_id: The identifier of the source node.
        """
        if from_id not in self.inbound_edges:
            self.inbound_edges.append(from_id)

    def add_outbound_edge(self, *, to_id: str) -> None:
        """Adds an outbound edge to the specified node.

        Args:
            to_id: The identifier of the destination node.
        """
        if to_id not in self.outbound_edges:
            self.outbound_edges.append(to_id)

    @property
    def attrs(self) -> XCSNodeAttributes:
        """Alias for attributes, for compatibility with tracing code."""
        return self.attributes


class XCSGraph(Generic[I, O]):
    """
    The canonical intermediate representation (IR) for XCS execution.

    XCSGraph is a directed acyclic graph (DAG) that defines the complete execution plan
    for a computation pipeline. It serves as the foundation for the XCS execution engine,
    providing a well-defined structure for operator composition, dependency tracking,
    and parallel execution scheduling.

    The graph maintains strict topological ordering guarantees and provides utilities
    for traversal, validation, and manipulation. This design enables powerful optimizations
    including operator fusion, parallel execution, and selective recomputation.

    Design principles:
    - Immutability after construction for reliable execution
    - Clean separation between graph structure and execution strategy
    - Explicit entry/exit points for deterministic data flow
    - Rich metadata for scheduling and optimization

    Attributes:
        nodes: Dictionary mapping node IDs to their XCSNode instances.
        entry_node: ID of the designated entry point for execution.
        exit_node: ID of the designated final node from which to collect results.
    """

    def __init__(self) -> None:
        """Initializes an empty XCSGraph."""
        self.nodes: Dict[str, XCSNode[I, O]] = {}
        self.entry_node: Optional[str] = None
        self.exit_node: Optional[str] = None

    def add_node(
        self, 
        operator: Callable[[I], O], 
        node_id: Optional[str] = None, 
        name: Optional[str] = None, 
        **attributes: object
    ) -> str:
        """
        Adds a new node to the graph.

        If node_id is not provided, a unique ID is generated. The first node added becomes the entry node,
        and each new node becomes the current exit node.

        Args:
            operator: The operator (callable) to attach to this node.
            node_id: Optional unique identifier for the node.
            name: Optional name for the node (used for compatibility with older code).
                If provided, it will be used as the node_id.
            **attributes: Additional attributes to attach to the node.

        Returns:
            The unique identifier for the newly added node.

        Raises:
            ValueError: If a node with the provided node_id already exists.
        """
        # For backward compatibility with APIs that use 'name' parameter
        if name is not None and node_id is None:
            node_id = name
            
        node = XCSNode(operator=operator, node_id=node_id)
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists.")
            
        # Add provided attributes
        for key, value in attributes.items():
            if key == "name":
                node.attributes["name"] = str(value)
            elif key == "description":
                node.attributes["description"] = str(value)
            elif key == "tags" and isinstance(value, list):
                node.attributes["tags"] = [str(tag) for tag in value]
            elif key == "metadata" and isinstance(value, dict):
                # Create a properly typed metadata dictionary from scratch
                typed_metadata: NodeMetadata = {}
                
                # Handle known fields
                source_file = value.get("source_file")
                if source_file is not None:
                    typed_metadata["source_file"] = str(source_file)
                    
                source_line = value.get("source_line")
                if source_line is not None and isinstance(source_line, int):
                    typed_metadata["source_line"] = source_line
                    
                author = value.get("author")
                if author is not None:
                    typed_metadata["author"] = str(author)
                    
                version = value.get("version")
                if version is not None:
                    typed_metadata["version"] = str(version)
                    
                created_at = value.get("created_at")
                if created_at is not None:
                    typed_metadata["created_at"] = str(created_at)
                    
                updated_at = value.get("updated_at")
                if updated_at is not None:
                    typed_metadata["updated_at"] = str(updated_at)
                    
                description = value.get("description")
                if description is not None:
                    typed_metadata["description"] = str(description)
                
                # Collect custom data for any other keys
                custom_data = value.get("custom_data", {})
                if isinstance(custom_data, dict) and custom_data:
                    typed_metadata["custom_data"] = dict(custom_data)
                node.attributes["metadata"] = typed_metadata
            else:
                # For any other attributes, store them in metadata.custom_data
                str_key = str(key)
                
                # Initialize metadata with custom_data
                if "metadata" not in node.attributes:
                    node.attributes["metadata"] = {"custom_data": {str_key: value}}
                else:
                    # Ensure there's a custom_data dictionary
                    metadata = node.attributes["metadata"]
                    if "custom_data" not in metadata:
                        metadata["custom_data"] = {str_key: value}
                    else:
                        # Add to existing custom_data - use indexing to satisfy type checker
                        metadata["custom_data"][str_key] = value
                
        self.nodes[node.node_id] = node

        if self.entry_node is None:
            self.entry_node = node.node_id
        self.exit_node = node.node_id

        return node.node_id

    def add_edge(self, from_id: str, to_id: Optional[str] = None, **kwargs: str) -> None:
        """Adds a directed edge from one node to another.

        Args:
            from_id: The source node's identifier.
            to_id: The destination node's identifier.
                If not provided, it must be specified via kwargs.
            **kwargs: Alternative way to provide source/destination IDs.

        Raises:
            ValueError: If either the source or destination node does not exist.
        """
        # Handle different calling patterns for backward compatibility
        if to_id is None:
            # Check if kwargs contains from_id and to_id
            if 'from_id' in kwargs and 'to_id' in kwargs:
                from_id = kwargs['from_id']
                to_id = kwargs['to_id']
            else:
                raise ValueError(
                    "Missing destination node ID. Use add_edge(from_id, to_id) "
                    "or add_edge(from_id=id1, to_id=id2)"
                )
                
        # Validate existence of nodes
        if from_id not in self.nodes:
            raise ValueError(f"Source node with ID '{from_id}' does not exist.")
        if to_id not in self.nodes:
            raise ValueError(f"Destination node with ID '{to_id}' does not exist.")
            
        # Add bidirectional references for the edge
        self.nodes[from_id].add_outbound_edge(to_id=to_id)
        self.nodes[to_id].add_inbound_edge(from_id=from_id)

    def get_node(self, node_id: Optional[str] = None, **kwargs: str) -> XCSNode[I, O]:
        """Retrieves the node with the specified identifier.

        Args:
            node_id: The identifier of the node to retrieve.
                If not provided, it must be specified via kwargs.
            **kwargs: Alternative way to provide node_id.

        Returns:
            The corresponding node instance.

        Raises:
            ValueError: If the node is not present in the graph.
        """
        # Handle different calling patterns for backward compatibility
        if node_id is None:
            if 'node_id' in kwargs:
                node_id = kwargs['node_id']
            else:
                raise ValueError("Missing node ID. Use get_node(node_id) or get_node(node_id=id)")
                
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' is not present in the graph.")
        return self.nodes[node_id]

    def all_node_ids(self) -> List[str]:
        """Returns a list of all node IDs present in the graph.

        Returns:
            A list containing all node identifiers.
        """
        return list(self.nodes.keys())

    def topological_sort(self) -> List[str]:
        """Performs a topological sort on the graph.

        This sorting ensures that for every directed edge from node A to node B, 
        node A appears before node B in the ordering.

        Returns:
            A list of node IDs in topologically sorted order.

        Raises:
            ValueError: If the graph contains a cycle or is otherwise not a valid DAG.
        """
        in_degree: Dict[str, int] = {
            node_id: len(node.inbound_edges) for node_id, node in self.nodes.items()
        }
        ready: List[str] = [
            node_id for node_id, degree in in_degree.items() if degree == 0
        ]
        sorted_list: List[str] = []

        while ready:
            current: str = ready.pop()
            sorted_list.append(current)
            for neighbor_id in self.nodes[current].outbound_edges:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    ready.append(neighbor_id)

        if len(sorted_list) != len(self.nodes):
            raise ValueError("Graph contains a cycle or is not a valid DAG.")

        return sorted_list
        
    def get_predecessors(self, node_id: str) -> Set[str]:
        """Get all immediate predecessors of a node.
        
        Args:
            node_id: ID of the node whose predecessors to find
            
        Returns:
            Set of node IDs that are immediate predecessors
            
        Raises:
            ValueError: If node with given ID doesn't exist
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' is not present in the graph.")
            
        return set(self.nodes[node_id].inbound_edges)
        
    def get_successors(self, node_id: str) -> Set[str]:
        """Get all immediate successors of a node.
        
        Args:
            node_id: ID of the node whose successors to find
            
        Returns:
            Set of node IDs that are immediate successors
            
        Raises:
            ValueError: If node with given ID doesn't exist
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' is not present in the graph.")
            
        return set(self.nodes[node_id].outbound_edges)
        
    def __iter__(self) -> Iterator[XCSNode[I, O]]:
        """Iterate over all nodes in the graph.
        
        Returns:
            Iterator over all nodes
        """
        return iter(self.nodes.values())


def merge_xcs_graphs(
    *, base: XCSGraph[I, O], additional: XCSGraph[I, O], namespace: Optional[str] = None
) -> XCSGraph[I, O]:
    """
    Merges two XCSGraph objects into a unified execution graph with proper namespacing.

    This function implements a graph composition operation that preserves the structural
    integrity and execution semantics of both input graphs. It automatically handles
    node ID conflicts through namespacing, ensuring that the resulting graph maintains
    a valid topological ordering and proper data flow between components.

    The composition connects the exit node of the additional graph to the entry node
    of the base graph, effectively creating a sequential pipeline where the additional
    graph executes first, followed by the base graph.

    This operation adheres to the Composite pattern, allowing complex graphs to be
    constructed through composition of simpler building blocks.

    Args:
        base: Primary execution graph that will incorporate the additional graph.
        additional: Secondary graph to be merged into the base graph.
        namespace: Optional prefix for node IDs from the additional graph
                   to prevent conflicts. If not provided, node IDs may be
                   automatically modified to avoid conflicts.

    Returns:
        A new XCSGraph containing all nodes from both input graphs with
        proper connectivity and namespacing.
    """
    ns_prefix: str = f"{namespace}_" if namespace else ""
    
    # Add all nodes from the additional graph to the base graph
    for node_id, node in additional.nodes.items():
        new_node_id: str = f"{ns_prefix}{node_id}"
        
        # Handle ID conflicts by appending a suffix
        while new_node_id in base.nodes:
            new_node_id += "_dup"
            
        # Update the node's ID
        node.node_id = new_node_id
        
        # Add to base graph
        base.nodes[new_node_id] = node
        
    # Connect the exit node of the additional graph to the entry node of the base graph
    if additional.exit_node:
        namespaced_exit: str = f"{ns_prefix}{additional.exit_node}"
        if base.exit_node:
            base.add_edge(from_id=namespaced_exit, to_id=base.exit_node)
        else:
            base.exit_node = namespaced_exit
            
    return base
