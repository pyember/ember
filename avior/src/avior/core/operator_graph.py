# operator_graph.py
import uuid
from typing import Any, Dict, List, Optional
from src.avior.registry.operator.operator_base import Operator


class OperatorGraphNode:
    """
    Represents a single node in the OperatorGraph. 
    It contains:
      - a unique node_id (string)
      - the operator reference (an instance of Operator)
      - inbound_edges: list of node_ids that feed into this node
      - outbound_edges: list of node_ids that receive output from this node
      - captured_inputs: optionally store the actual input dict used during an execution/tracing
      - captured_outputs: optionally store the output from the operator
    """

    def __init__(self, 
                 node_id: str,
                 operator: Operator):
        self.node_id: str = node_id
        self.operator: Operator = operator
        self.inbound_edges: List[str] = []
        self.outbound_edges: List[str] = []
        self.captured_inputs: Dict[str, Any] = {}
        self.captured_outputs: Any = None

    def add_inbound_edge(self, from_id: str) -> None:
        if from_id not in self.inbound_edges:
            self.inbound_edges.append(from_id)

    def add_outbound_edge(self, to_id: str) -> None:
        if to_id not in self.outbound_edges:
            self.outbound_edges.append(to_id)


class OperatorGraph:
    """
    A unified internal representation of a DAG of Operators. 
    This replaces prior NoNGraphData, GraphNode, TracedGraph, etc. 
    """

    def __init__(self):
        self.nodes: Dict[str, OperatorGraphNode] = {}
        self.entry_node: Optional[str] = None
        self.exit_node: Optional[str] = None

    def add_node(self, operator: Operator, node_id: Optional[str] = None) -> str:
        """
        Creates a new node in the graph with the provided operator, 
        auto-generating a unique ID if node_id is not given.
        
        Returns:
            The node_id of the newly added node.
        """
        if not node_id:
            node_id = str(uuid.uuid4())[:8]  # short random ID

        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in the graph.")

        node = OperatorGraphNode(node_id=node_id, operator=operator)
        self.nodes[node_id] = node

        if self.entry_node is None:
            self.entry_node = node_id
        # By default, each newly added node can become the 'exit' 
        # (the last inserted node is the 'exit' unless changed).
        self.exit_node = node_id

        return node_id

    def add_edge(self, from_id: str, to_id: str) -> None:
        """
        Connects two nodes in the graph. 
        from_id => to_id
        """
        if from_id not in self.nodes:
            raise ValueError(f"Node {from_id} does not exist.")
        if to_id not in self.nodes:
            raise ValueError(f"Node {to_id} does not exist.")

        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]

        from_node.add_outbound_edge(to_id)
        to_node.add_inbound_edge(from_id)

    def get_node(self, node_id: str) -> OperatorGraphNode:
        """
        Retrieve a node by its ID.
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist.")
        return self.nodes[node_id]

    def all_node_ids(self) -> List[str]:
        """
        Return a list of all node IDs in the graph.
        """
        return list(self.nodes.keys())