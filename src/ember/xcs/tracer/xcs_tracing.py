"""
Unified Tracing Module for XCS.

Provides the TracerContext for capturing sub-operator calls during execution
and converting a traced graph into an execution plan.
"""

import logging
from functools import wraps
from types import TracebackType
from typing import Any, Callable, Dict, Optional, Protocol, Type

from ..graph.xcs_graph import XCSGraph

logger = logging.getLogger(__name__)


class OperatorCallable(Protocol):
    """Protocol for an operator callable accepting a named 'inputs' argument.

    This protocol ensures that an operator’s __call__ method accepts an 'inputs'
    keyword argument of type Dict[str, Any] and returns an arbitrary result.
    """

    def __call__(self, *, inputs: Dict[str, Any]) -> Any: ...


class TracerContext:
    """Captures operator calls during execution to build a traced XCSGraph.

    The TracerContext patches the __call__ method of the provided top-level operator
    to record execution traces. The captured information is accumulated in an internal
    XCSGraph, which can later be converted to an executable plan.
    """

    def __init__(
        self, top_operator: OperatorCallable, sample_input: Dict[str, Any]
    ) -> None:
        """Initializes a TracerContext instance.

        Args:
            top_operator (OperatorCallable): The top-level operator to be traced.
            sample_input (Dict[str, Any]): A sample input dictionary used for tracing.
        """
        self.top_operator: OperatorCallable = top_operator
        self.sample_input: Dict[str, Any] = sample_input
        self.graph: XCSGraph = XCSGraph()
        self._patched: bool = False
        self._original_call: Optional[Callable[..., Any]] = getattr(
            top_operator, "__call__", None
        )

    def __enter__(self) -> "TracerContext":
        """Enters the tracing context by patching the operator's __call__ method.

        Returns:
            TracerContext: This context instance.
        """
        self._patch_operator(op=self.top_operator)
        self._patched = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exits the tracing context and restores the original operator __call__ method.

        Args:
            exc_type (Optional[Type[BaseException]]): The exception type, if any.
            exc_value (Optional[BaseException]): The exception instance, if any.
            exc_tb (Optional[TracebackType]): The traceback, if any.

        Returns:
            Optional[bool]: Always returns None.
        """
        if self._patched and self._original_call is not None:
            self.top_operator.__call__ = self._original_call
        self._patched = False
        return None

    def _patch_operator(self, op: OperatorCallable) -> None:
        """Patches the operator's __call__ method to record tracing data.

        The patched __call__ method creates a node in the tracing graph, invokes the
        original __call__ with the given inputs, captures its outputs in the graph node,
        and returns the result.

        Args:
            op (OperatorCallable): The operator whose __call__ method is to be patched.
        """
        original_call = op.__call__

        @wraps(original_call)
        def wrapped_call(*, inputs: Dict[str, Any]) -> Any:
            # Create a node in the tracing graph.
            node_id: str = self.graph.add_node(operator=op)
            result: Any = original_call(inputs=inputs)
            self.graph.get_node(node_id).captured_outputs = result
            return result

        op.__call__ = wrapped_call

    def run_trace(self) -> XCSGraph:
        """Executes the top operator with the sample input to populate the tracing graph.

        Returns:
            XCSGraph: The traced graph populated during operator execution.
        """
        self.top_operator(inputs=self.sample_input)
        return self.graph


def convert_traced_graph_to_plan(*, tracer_graph: XCSGraph) -> Any:
    """Converts a traced XCSGraph into an executable plan.

    This function iterates over all nodes in the traced graph, creates corresponding
    execution plan tasks, and aggregates them into a comprehensive execution plan.

    Args:
        tracer_graph (XCSGraph): The traced XCSGraph instance.

    Returns:
        Any: The execution plan derived from the traced graph.
    """
    # Import here to avoid circular dependency issues.
    from ..engine.xcs_engine import XCSPlan, XCSPlanTask

    plan: XCSPlan = XCSPlan()
    for node_id, node in tracer_graph.nodes.items():
        task: XCSPlanTask = XCSPlanTask(
            node_id=node_id, operator=node.operator, inbound_nodes=node.inbound_edges
        )
        plan.add_task(task)
    return plan
