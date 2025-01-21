import uuid
from typing import Any, Dict, List, Tuple, Optional

from src.avior.registry.operator.operator_base import Operator
from src.avior.core.scheduler import ExecutionPlan, ExecutionTask

class TracedNode:
    """
    Represents a single sub-operator call in the traced graph.
    Could store:
      - name or ID
      - operator reference
      - inputs (or references to upstream node IDs)
      - outputs
    """
    def __init__(self, node_id: str, operator: Operator):
        self.node_id = node_id
        self.operator = operator
        self.inbound_edges: List[str] = []   # node_ids of parents
        self.outbound_edges: List[str] = []  # node_ids of children
        self.inputs: Dict[str, Any] = {}     # captured inputs
        self.outputs: Any = None            # captured outputs

class TracedGraph:
    """
    Holds a collection of TracedNode objects plus adjacency.
    This is a pure data structure representing the call DAG discovered by the tracer.
    """
    def __init__(self):
        self.nodes: Dict[str, TracedNode] = {}
        self.entry_node: Optional[str] = None
        self.exit_node: Optional[str] = None

    def add_node(self, node: TracedNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists.")
        if self.entry_node is None:
            self.entry_node = node.node_id
        self.exit_node = node.node_id
        self.nodes[node.node_id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        self.nodes[from_id].outbound_edges.append(to_id)
        self.nodes[to_id].inbound_edges.append(from_id)

class TraceCallInterceptor:
    """
    Intercepts calls to sub-operators, capturing input->output edges.
    We'll monkey-patch each sub-operator's __call__ during the trace.
    """

    def __init__(self, top_operator: Operator, tracer_graph: TracedGraph):
        self.top_operator = top_operator
        self.tracer_graph = tracer_graph
        self.original_calls = {}  # store original __call__ methods

    def instrument(self):
        """
        Recursively instrument the top_operator and its sub_operators 
        so that each call is captured.
        """
        self._walk_and_instrument(self.top_operator)

    def restore(self):
        """
        Restore original __call__ references after tracing.
        """
        for op, original_call in self.original_calls.items():
            op.__call__ = original_call

    def _walk_and_instrument(self, operator: Operator, path: str = "", visited=None):
        if visited is None:
            visited = set()

        # Skip instrumenting the top operator itself to avoid infinite self-calls:
        if operator is self.top_operator:
            # Still recurse on sub-operators.
            for sub_name, sub_op in operator.sub_operators.items():
                sub_path = f"{path}.{sub_name}" if path else sub_name
                self._walk_and_instrument(sub_op, sub_path, visited)
            return

        if operator in visited:
            return
        visited.add(operator)

        if operator not in self.original_calls:
            self._instrument_operator(operator, path)

        for sub_name, sub_op in operator.sub_operators.items():
            sub_path = f"{path}.{sub_name}" if path else sub_name
            self._walk_and_instrument(sub_op, sub_path, visited)

    def _instrument_operator(self, operator: Operator, path: str) -> None:
        # Ensure we do not re-instrument an already-patched operator
        if getattr(operator, "_already_instrumented", False):
            return
        setattr(operator, "_already_instrumented", True)

        original_call = operator.__call__

        def wrapped_call(inputs):
            # 1) Create a node representing this call
            node_id = f"{path or 'root'}_{str(uuid.uuid4())[:8]}"
            node = TracedNode(node_id=node_id, operator=operator)
            node.inputs = inputs  # or copy

            # 2) Actually call the original operator
            outputs = original_call(inputs)

            # 3) Capture outputs in the node
            node.outputs = outputs

            # 4) Add node to tracer_graph
            self.tracer_graph.add_node(node)

            # 5) For dataflow edges, we link from previously active node(s) to this node
            #    We might keep track of a "current active node" in the tracer context,
            #    or store in thread-local variable.
            #    For simplicity, we won't show the full approach here, but conceptually:
            #    if there's a "parent_node" tracked, do tracer_graph.add_edge(parent_id, node_id)

            return outputs

        self.original_calls[operator] = original_call
        operator.__call__ = wrapped_call

class TracerContext:
    """
    The top-level context manager or decorator that:
      1. Creates a TracedGraph
      2. Instruments the top_operator
      3. Calls the operator forward with sample input
      4. Restores the original calls
    """

    def __init__(self, top_operator: Operator, sample_input: Any):
        self.top_operator = top_operator
        self.sample_input = sample_input
        self.tracer_graph = TracedGraph()
        self.interceptor = TraceCallInterceptor(top_operator, self.tracer_graph)

    def __enter__(self):
        self.interceptor.instrument()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.interceptor.restore()

    def run_trace(self):
        """
        Runs the actual forward pass, capturing the graph.
        """
        _ = self.top_operator(self.sample_input)
        # The tracer_graph is now populated
        return self.tracer_graph

def convert_traced_graph_to_plan(traced_graph: TracedGraph) -> ExecutionPlan:
    plan = ExecutionPlan()
    # We can do a topological sort over traced_graph.nodes by inbound_edges
    # Then each node => ExecutionTask
    # For example:
    for node_id, node in traced_graph.nodes.items():
        # A single task calls the operator in eager style with the inputs 
        # (Though in real HPC usage, we might do partial plan-based concurrency).
        def task_fn(operator=node.operator, inps=node.inputs):
            return operator.forward(inps)
        plan.add_task(ExecutionTask(
            task_id=node_id,
            function=task_fn,
            inputs={},  # if needed
            dependencies=node.inbound_edges
        ))
    return plan