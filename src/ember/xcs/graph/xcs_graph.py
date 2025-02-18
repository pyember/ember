import uuid
import logging
from typing import Any, Dict, List, Optional

logger: logging.Logger = logging.getLogger(__name__)


class XCSNode:
    """
    Represents a single node in the unified XCS graph IR.

    This node encapsulates an operator along with its connectivity data. In addition,
    each node carries an 'attributes' dictionary for scheduling hints (e.g. prompt length)
    and other metadata.

    Attributes:
        node_id (str): A unique identifier for the node.
        operator (Any): The operator (callable) associated with the node.
        inbound_edges (List[str]): IDs of nodes whose outputs feed into this node.
        outbound_edges (List[str]): IDs of nodes that consume this node's output.
        attributes (Dict[str, Any]): Extra metadata (e.g., scheduling hints).
        captured_inputs (Dict[str, Any]): Debug or execution introspection inputs.
        captured_outputs (Optional[Any]): The outputs produced by the operator (if executed).
    """

    def __init__(self, operator: Any, node_id: Optional[str] = None) -> None:
        """Initializes an XCSNode instance.

        Args:
            operator (Any): The operator callable for this node.
            node_id (Optional[str]): An optional node identifier; if not provided, a new unique ID is generated.
        """
        if node_id is None:
            node_id = str(uuid.uuid4())[:8]
        self.node_id: str = node_id
        self.operator: Any = operator
        self.inbound_edges: List[str] = []
        self.outbound_edges: List[str] = []
        self.attributes: Dict[str, Any] = {}
        self.captured_inputs: Dict[str, Any] = {}
        self.captured_outputs: Optional[Any] = None

    def add_inbound_edge(self, *, from_id: str) -> None:
        """Adds an inbound edge from the specified node.

        Args:
            from_id (str): The identifier of the source node.
        """
        if from_id not in self.inbound_edges:
            self.inbound_edges.append(from_id)

    def add_outbound_edge(self, *, to_id: str) -> None:
        """Adds an outbound edge to the specified node.

        Args:
            to_id (str): The identifier of the destination node.
        """
        if to_id not in self.outbound_edges:
            self.outbound_edges.append(to_id)

    @property
    def attrs(self) -> Dict[str, Any]:
        """Alias for attributes, for compatibility with tracing code."""
        return self.attributes


class XCSGraph:
    """
    The canonical intermediate representation (IR) for XCS.

    This directed acyclic graph (DAG) defines operator composition and scheduling.
    Each node in the graph corresponds to a lowered operator invocation and carries attributes
    that provide scheduling hints for the execution engine.

    Attributes:
        nodes (Dict[str, XCSNode]): Mapping from node IDs to XCSNode instances.
        entry_node (Optional[str]): The ID of the "entry" node.
        exit_node (Optional[str]): The ID of the "exit" node.
    """

    def __init__(self) -> None:
        """Initializes an empty XCSGraph."""
        self.nodes: Dict[str, XCSNode] = {}
        self.entry_node: Optional[str] = None
        self.exit_node: Optional[str] = None

    def add_node(self, *, operator: Any, node_id: Optional[str] = None) -> str:
        """
        Adds a new node to the graph.

        If node_id is not provided, a unique ID is generated. The first node added becomes the entry node,
        and each new node becomes the current exit node.

        Args:
            operator (Any): The operator (callable) to attach to this node.
            node_id (Optional[str]): Optional unique identifier for the node.

        Returns:
            str: The unique identifier for the newly added node.

        Raises:
            ValueError: If a node with the provided node_id already exists.
        """
        node: XCSNode = XCSNode(operator=operator, node_id=node_id)
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists.")
        self.nodes[node.node_id] = node

        if self.entry_node is None:
            self.entry_node = node.node_id
        self.exit_node = node.node_id

        return node.node_id

    def add_edge(self, *, from_id: str, to_id: str) -> None:
        """Adds a directed edge from one node to another.

        Args:
            from_id (str): The source node's identifier.
            to_id (str): The destination node's identifier.

        Raises:
            ValueError: If either the source or destination node does not exist.
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node with ID '{from_id}' does not exist.")
        if to_id not in self.nodes:
            raise ValueError(f"Destination node with ID '{to_id}' does not exist.")
        self.nodes[from_id].add_outbound_edge(to_id=to_id)
        self.nodes[to_id].add_inbound_edge(from_id=from_id)

    def get_node(self, *, node_id: str) -> XCSNode:
        """Retrieves the node with the specified identifier.

        Args:
            node_id (str): The identifier of the node to retrieve.

        Returns:
            XCSNode: The corresponding node instance.

        Raises:
            ValueError: If the node is not present in the graph.
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' is not present in the graph.")
        return self.nodes[node_id]

    def all_node_ids(self) -> List[str]:
        """Returns a list of all node IDs present in the graph.

        Returns:
            List[str]: A list containing all node identifiers.
        """
        return list(self.nodes.keys())

    def topological_sort(self) -> List[str]:
        """Performs a topological sort on the graph.

        This sorting ensures that for every directed edge from node A to node B, node A appears before node B in the ordering.

        Returns:
            List[str]: A list of node IDs in topologically sorted order.

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


# Helper function to merge two XCSGraphs.
def merge_xcs_graphs(
    *, base: XCSGraph, additional: XCSGraph, namespace: Optional[str] = None
) -> XCSGraph:
    """
    Merge two XCSGraph objects into one unified execution graph with namespaced node IDs.

    Args:
        base (XCSGraph): The base execution graph.
        additional (XCSGraph): The graph to merge into the base.
        namespace (Optional[str]): Optional namespace prefix for node IDs.

    Returns:
        XCSGraph: The merged execution graph.
    """
    ns_prefix: str = f"{namespace}_" if namespace else ""
    for node_id, node in additional.nodes.items():
        new_node_id: str = f"{ns_prefix}{node_id}"
        while new_node_id in base.nodes:
            new_node_id += "_dup"
        node.node_id = new_node_id
        base.nodes[new_node_id] = node
    if additional.exit_node:
        namespaced_exit: str = f"{ns_prefix}{additional.exit_node}"
        if base.exit_node:
            base.add_edge(from_id=namespaced_exit, to_id=base.exit_node)
        else:
            base.exit_node = namespaced_exit
    return base
