import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Type

from src.ember.registry.operator.operator_base import Operator
from src.ember.core.scheduler import ExecutionPlan, ExecutionTask


class TracedNode:
    """Represents a single sub-operator call in the traced graph.

    Attributes:
        node_id (str): Unique identifier for the node.
        operator (Operator): The operator instance associated with this node.
        inbound_edges (List[str]): List of parent node IDs.
        outbound_edges (List[str]): List of child node IDs.
        inputs (Dict[str, Any]): Captured input data for the operator.
        outputs (Any): Captured output from the operator.
    """

    def __init__(self, *, node_id: str, operator: Operator) -> None:
        """Initializes a TracedNode.

        Args:
            node_id (str): A unique string identifier for the node.
            operator (Operator): The operator associated with this node.
        """
        self.node_id: str = node_id
        self.operator: Operator = operator
        self.inbound_edges: List[str] = []
        self.outbound_edges: List[str] = []
        self.inputs: Dict[str, Any] = {}
        self.outputs: Any = None


class TracedGraph:
    """Represents a traced call directed acyclic graph (DAG) composed of TracedNode objects.

    Attributes:
        nodes (Dict[str, TracedNode]): Mapping from node IDs to TracedNode instances.
        entry_node (Optional[str]): Node ID of the entry node in the DAG.
        exit_node (Optional[str]): Node ID of the most recently added (exit) node.
    """

    def __init__(self) -> None:
        """Initializes an empty TracedGraph."""
        self.nodes: Dict[str, TracedNode] = {}
        self.entry_node: Optional[str] = None
        self.exit_node: Optional[str] = None

    def add_node(self, *, node: TracedNode) -> None:
        """Adds a node to the traced graph.

        Args:
            node (TracedNode): The TracedNode instance to add.

        Raises:
            ValueError: If a node with the same node_id already exists.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists.")
        if self.entry_node is None:
            self.entry_node = node.node_id
        self.exit_node = node.node_id
        self.nodes[node.node_id] = node

    def add_edge(self, *, from_id: str, to_id: str) -> None:
        """Adds an edge between two nodes in the traced graph.

        Args:
            from_id (str): The node ID where the edge originates.
            to_id (str): The node ID where the edge terminates.
        """
        self.nodes[from_id].outbound_edges.append(to_id)
        self.nodes[to_id].inbound_edges.append(from_id)


class TraceCallInterceptor:
    """Intercepts calls to sub-operators by monkey-patching __call__ to capture input/output details.

    Attributes:
        top_operator (Operator): The top-level operator to begin instrumentation.
        tracer_graph (TracedGraph): The graph to record traced calls.
        original_calls (Dict[Operator, Callable[[Dict[str, Any]], Any]]):
            Mapping from instrumented operators to their original __call__ methods.
    """

    def __init__(self, *, top_operator: Operator, tracer_graph: TracedGraph) -> None:
        """Initializes the TraceCallInterceptor.

        Args:
            top_operator (Operator): The top-level operator to instrument.
            tracer_graph (TracedGraph): The graph to record traced calls.
        """
        self.top_operator: Operator = top_operator
        self.tracer_graph: TracedGraph = tracer_graph
        self.original_calls: Dict[Operator, Callable[[Dict[str, Any]], Any]] = {}

    def instrument(self) -> None:
        """Recursively instruments the top operator and all its sub-operators."""
        self._walk_and_instrument(operator=self.top_operator)

    def restore(self) -> None:
        """Restores the original __call__ methods for all instrumented operators."""
        for op, original_call in self.original_calls.items():
            op.__call__ = original_call

    def _walk_and_instrument(
        self,
        *,
        operator: Operator,
        path: str = "",
        visited: Optional[Set[Operator]] = None,
    ) -> None:
        """Recursively traverses the operator tree and instruments each sub-operator.

        Args:
            operator (Operator): The current operator to instrument.
            path (str): Dot-separated string specifying the operator's hierarchical location.
            visited (Optional[Set[Operator]]): Set of operators already visited to avoid duplication.
        """
        if visited is None:
            visited = set()

        # Avoid instrumenting the top operator itself to prevent infinite recursion.
        if operator is self.top_operator:
            for sub_name, sub_op in operator.sub_operators.items():
                sub_path: str = f"{path}.{sub_name}" if path else sub_name
                self._walk_and_instrument(
                    operator=sub_op, path=sub_path, visited=visited
                )
            return

        if operator in visited:
            return
        visited.add(operator)

        if operator not in self.original_calls:
            self._instrument_operator(operator=operator, path=path)

        for sub_name, sub_op in operator.sub_operators.items():
            sub_path: str = f"{path}.{sub_name}" if path else sub_name
            self._walk_and_instrument(operator=sub_op, path=sub_path, visited=visited)

    def _instrument_operator(self, *, operator: Operator, path: str) -> None:
        """Monkey-patches the operator's __call__ method to capture its input/output behavior.

        Args:
            operator (Operator): The operator instance to patch.
            path (str): Hierarchical path identifying the operator.
        """
        if getattr(operator, "_already_instrumented", False):
            return
        setattr(operator, "_already_instrumented", True)

        original_call: Callable[[Dict[str, Any]], Any] = operator.__call__

        def wrapped_call(*, inputs: Dict[str, Any]) -> Any:
            """Wrapped version of __call__ that records call details.

            Args:
                inputs (Dict[str, Any]): Dictionary containing input data for the operator.

            Returns:
                Any: The output produced by the original operator call.
            """
            node_id: str = f"{path or 'root'}_{str(uuid.uuid4())[:8]}"
            node: TracedNode = TracedNode(node_id=node_id, operator=operator)
            node.inputs = inputs

            # Invoke the original __call__ method using named invocation.
            outputs: Any = original_call(inputs=inputs)
            node.outputs = outputs

            self.tracer_graph.add_node(node=node)
            return outputs

        self.original_calls[operator] = original_call
        operator.__call__ = wrapped_call


class TracerContext:
    """Context manager for tracing operator calls and building a traced call graph.

    Attributes:
        top_operator (Operator): The top-level operator to be traced.
        sample_input (Any): Sample input data for executing the operator.
        tracer_graph (TracedGraph): The graph that records operator call traces.
        interceptor (TraceCallInterceptor): Instrumentation manager for capturing operator calls.
    """

    def __init__(self, *, top_operator: Operator, sample_input: Any) -> None:
        """Initializes a TracerContext.

        Args:
            top_operator (Operator): The operator whose calls are to be traced.
            sample_input (Any): Sample input data used during execution.
        """
        self.top_operator: Operator = top_operator
        self.sample_input: Any = sample_input
        self.tracer_graph: TracedGraph = TracedGraph()
        self.interceptor: TraceCallInterceptor = TraceCallInterceptor(
            top_operator=self.top_operator, tracer_graph=self.tracer_graph
        )

    def __enter__(self) -> "TracerContext":
        """Enters the tracing context and instruments the operator.

        Returns:
            TracerContext: The current tracing context instance.
        """
        self.interceptor.instrument()
        return self

    def __exit__(
        self,
        *,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exits the tracing context and restores original operator behavior.

        Args:
            exc_type (Optional[Type[BaseException]]): The exception class, if an exception occurred.
            exc_val (Optional[BaseException]): The exception instance, if an exception occurred.
            exc_tb (Optional[Any]): The traceback, if an exception occurred.
        """
        self.interceptor.restore()

    def run_trace(self) -> TracedGraph:
        """Executes the top operator to capture the trace of its calls.

        Returns:
            TracedGraph: The populated traced graph containing all recorded nodes.
        """
        _ = self.top_operator(inputs=self.sample_input)
        return self.tracer_graph


def convert_traced_graph_to_plan(*, traced_graph: TracedGraph) -> ExecutionPlan:
    """Converts a traced graph into an executable plan.

    Args:
        traced_graph (TracedGraph): The traced graph to convert.

    Returns:
        ExecutionPlan: An execution plan representing the traced operator calls.
    """
    plan: ExecutionPlan = ExecutionPlan()
    for node_id, node in traced_graph.nodes.items():

        def task_fn(
            *, operator: Operator = node.operator, inputs: Dict[str, Any] = node.inputs
        ) -> Any:
            """Executes the operator's forward pass.

            Args:
                operator (Operator): The operator instance.
                inputs (Dict[str, Any]): The input data for the operator.

            Returns:
                Any: The result of executing the operator's forward method.
            """
            return operator.forward(inputs=inputs)

        plan.add_task(
            task=ExecutionTask(
                task_id=node_id,
                function=task_fn,
                inputs={},  # Additional inputs can be provided here if needed.
                dependencies=node.inbound_edges,
            )
        )
    return plan
