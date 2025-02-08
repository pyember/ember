# INSERT_YOUR_REWRITE_HERE
import uuid
from typing import Any, Dict, List, Optional

from ember.src.ember.registry.operator.core.operator_base import Operator


class OperatorGraphNode:
    """Represents a node within an operator graph.

    This class encapsulates details for a single node including its unique
    identifier, associated operator instance, edges to other nodes, and
    captured execution I/O.

    Attributes:
        node_id (str): Unique identifier for the node.
        operator (Operator): The operator instance associated with this node.
        inbound_edges (List[str]): List of node IDs providing input to this node.
        outbound_edges (List[str]): List of node IDs receiving output from this node.
        captured_inputs (Dict[str, Any]): Input data captured during execution.
        captured_outputs (Optional[Any]): Output produced by the operator; initially None.
    """

    def __init__(self, *, node_id: str, operator: Operator) -> None:
        """Initializes an OperatorGraphNode with a unique identifier and an operator.

        Args:
            node_id (str): A unique identifier for the node.
            operator (Operator): The operator instance for this node.
        """
        self.node_id: str = node_id
        self.operator: Operator = operator
        self.inbound_edges: List[str] = []
        self.outbound_edges: List[str] = []
        self.captured_inputs: Dict[str, Any] = {}
        self.captured_outputs: Optional[Any] = None

    def add_inbound_edge(self, *, from_id: str) -> None:
        """Adds an inbound edge from a source node.

        Args:
            from_id (str): Identifier of the source node.
        """
        if from_id not in self.inbound_edges:
            self.inbound_edges.append(from_id)

    def add_outbound_edge(self, *, to_id: str) -> None:
        """Adds an outbound edge to a destination node.

        Args:
            to_id (str): Identifier of the destination node.
        """
        if to_id not in self.outbound_edges:
            self.outbound_edges.append(to_id)


class OperatorGraph:
    """Represents a directed acyclic graph (DAG) of operator nodes.

    This graph is a unified internal model that replaces legacy structures
    such as NoNGraphData, GraphNode, and TracedGraph.

    Attributes:
        nodes (Dict[str, OperatorGraphNode]): Mapping from node IDs to operator nodes.
        entry_node (Optional[str]): Node ID representing the entry point of the graph.
        exit_node (Optional[str]): Node ID representing the exit point of the graph.
    """

    def __init__(self) -> None:
        """Initializes an empty OperatorGraph."""
        self.nodes: Dict[str, OperatorGraphNode] = {}
        self.entry_node: Optional[str] = None
        self.exit_node: Optional[str] = None

    def add_node(self, *, operator: Operator, node_id: Optional[str] = None) -> str:
        """Adds a new node with the specified operator to the graph.

        If no node_id is provided, a unique short identifier is generated.
        The first node added is set as the entry node by default. Each new node
        becomes the exit node.

        Args:
            operator (Operator): The operator instance for the new node.
            node_id (Optional[str]): A unique identifier for the node; if None, one is generated.

        Returns:
            str: The unique identifier of the newly added node.

        Raises:
            ValueError: If a node with the provided node_id already exists.
        """
        if node_id is None:
            node_id = str(uuid.uuid4())[:8]

        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists in the graph.")

        node_instance: OperatorGraphNode = OperatorGraphNode(
            node_id=node_id, operator=operator
        )
        self.nodes[node_id] = node_instance

        if self.entry_node is None:
            self.entry_node = node_id

        # By design, the most recently added node becomes the exit node.
        self.exit_node = node_id

        return node_id

    def add_edge(self, *, from_id: str, to_id: str) -> None:
        """Connects two nodes by adding an edge from the source node to the destination node.

        Args:
            from_id (str): Identifier of the source node.
            to_id (str): Identifier of the destination node.

        Raises:
            ValueError: If the source or destination node does not exist.
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node '{from_id}' does not exist in the graph.")
        if to_id not in self.nodes:
            raise ValueError(f"Destination node '{to_id}' does not exist in the graph.")

        source_node: OperatorGraphNode = self.nodes[from_id]
        destination_node: OperatorGraphNode = self.nodes[to_id]

        source_node.add_outbound_edge(to_id=to_id)
        destination_node.add_inbound_edge(from_id=from_id)

    def get_node(self, *, node_id: str) -> OperatorGraphNode:
        """Retrieves a node from the graph by its unique identifier.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            OperatorGraphNode: The node corresponding to the provided identifier.

        Raises:
            ValueError: If no node with the specified identifier exists.
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' does not exist in the graph.")

        return self.nodes[node_id]

    def all_node_ids(self) -> List[str]:
        """Retrieves a list of all node identifiers in the graph.

        Returns:
            List[str]: A list containing all node IDs.
        """
        return list(self.nodes.keys())
