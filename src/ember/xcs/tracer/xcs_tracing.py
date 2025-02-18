"""
Unified Tracing Module for XCS.

This module provides a TracerContext for capturing operator calls
and constructing an XCSGraph-based execution IR. It patches operator calls,
records trace events, and then restores the originals.
"""

from __future__ import annotations

import logging
import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, Dict, Optional, Protocol, cast

from pydantic import BaseModel, Field

from src.ember.xcs.tracer.tracer_decorator import IRGraph
from src.ember.xcs.tracer._context_types import TraceContextData
from src.ember.xcs.utils.tree_util import tree_flatten, tree_unflatten

logger: logging.Logger = logging.getLogger(__name__)


class OperatorCallable(Protocol):
    """
    Protocol for an operator callable.

    An operator must implement a __call__ method accepting a keyword-only
    argument 'inputs'.
    """

    def __call__(self, *, inputs: Dict[str, Any]) -> Any: ...


class TraceRecord(BaseModel):
    """
    Represents a trace record for a single operator invocation.

    Stores the operator name, the node ID in the traced graph, the inputs and outputs,
    and a timestamp.
    """

    operator_name: str
    node_id: str
    inputs: Dict[str, Any]
    outputs: Any
    timestamp: float = Field(default_factory=time.time)


class TracerContext(ContextDecorator):
    """
    Context manager to trace operator execution and build an XCSGraph.

    This context patched the class-level __call__ methods of operator instances so that
    each operator call is recorded as a node in the graph and a corresponding TraceRecord
    is generated. Patching is done recursively for sub-operators.
    """

    def __init__(
        self,
        top_operator: OperatorCallable,
        sample_input: Dict[str, Any],
        extra_context: Optional[TraceContextData] = None,
    ) -> None:
        """
        Args:
            top_operator (OperatorCallable): The operator to trace.
            sample_input (Dict[str, Any]): Sample input for tracing.
            extra_context (Optional[TraceContextData]): Additional context info.
        """
        self.top_operator = top_operator
        self.sample_input = sample_input
        self.extra_context = extra_context
        self.graph = IRGraph()
        self._input_tracers: Dict[str, str] = {}
        # Keep track of which operator instances we patched.
        self._patched: set[OperatorCallable] = set()
        self.trace_records: list[TraceRecord] = []

    def __enter__(self) -> TracerContext:
        # Flatten the sample_input so each leaf becomes a node.
        leaves, aux = tree_flatten(tree=self.sample_input)
        for i, leaf in enumerate(leaves):
            node_id = self.graph.add_node(
                operator=None,
                inputs=[],
                attrs={"value": leaf, "index": i},
            )
            key = f"input_{i}"
            self.graph.input_mapping[key] = node_id
            self._input_tracers[key] = node_id

        self._patch_operator(self.top_operator)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        self._restore_operators()
        return None

    def _patch_operator(self, op: OperatorCallable) -> None:
        """
        Recursively patch an operator's class-level __call__ so that calls produce trace records.
        """
        # Avoid re-patching an operator instance.
        if getattr(op, "_tracer_patched", False):
            return

        cls = op.__class__
        # If this class hasn't been patched yet, save its original call.
        if not hasattr(cls, "_tracer_original_call"):
            setattr(cls, "_tracer_original_call", cls.__call__)

        original_call = cls.__call__

        @wraps(original_call)
        def patched_call(this, *, inputs: Dict[str, Any]) -> Any:
            operator_name = getattr(this, "name", this.__class__.__name__)
            node_id = self.graph.add_node(operator=this, inputs=[])
            node = self.graph.nodes[node_id]
            node.attrs["name"] = operator_name

            # Recursively patch sub-operators.
            if hasattr(this, "sub_operators") and isinstance(this.sub_operators, dict):
                for sub_op in this.sub_operators.values():
                    self._patch_operator(sub_op)

            result = original_call(this, inputs=inputs)
            node.captured_inputs = inputs
            node.captured_outputs = result
            self.trace_records.append(
                TraceRecord(
                    operator_name=operator_name,
                    node_id=node_id,
                    inputs=inputs,
                    outputs=result,
                )
            )
            return result

        cls.__call__ = patched_call
        object.__setattr__(op, "_tracer_patched", True)
        self._patched.add(op)

    def _restore_operators(self) -> None:
        """
        Restore each operator's class to its original call method.
        """
        for op in self._patched:
            cls = op.__class__
            if hasattr(cls, "_tracer_original_call"):
                cls.__call__ = getattr(cls, "_tracer_original_call")
                delattr(cls, "_tracer_original_call")
            if hasattr(op, "_tracer_patched"):
                delattr(op, "_tracer_patched")
        self._patched.clear()

    def run_trace(self) -> IRGraph:
        """
        Executes the top operator with the sample input to build the trace.

        Returns:
            IRGraph: The constructed execution graph.
        """
        # Reconstruct inputs as a PyTree using the tracer references.
        leaves, aux = tree_flatten(tree=self.sample_input)
        unflattened_inputs = tree_unflatten(aux=aux, children=leaves)

        # Add the operator node
        operator_node_id = self.graph.add_node(
            operator=self.top_operator,
            inputs=list(self._input_tracers.values()),
        )

        # Invoke the operator with reconstructed inputs
        output = self.top_operator(inputs=cast(Dict[str, Any], unflattened_inputs))

        # Flatten and record the output leaves
        out_leaves, out_aux = tree_flatten(tree=output)
        # Ensure the operator node has an 'attrs' attribute (for cases where it is a XCSNode)
        if not hasattr(self.graph.nodes[operator_node_id], "attrs"):
            self.graph.nodes[operator_node_id].attrs = {}
        for i, leaf in enumerate(out_leaves):
            self.graph.output_mapping[f"output_{i}"] = operator_node_id
            # Store the output leaves in the operator node for reference
            self.graph.nodes[operator_node_id].attrs[f"output_{i}"] = leaf

        logger.debug("Constructed IR graph with %d nodes.", len(self.graph.nodes))
        return self.graph


def convert_traced_graph_to_plan(*, tracer_graph: IRGraph) -> Any:
    """
    Converts a traced XCSGraph into an executable plan (XCSPlan).
    """
    from src.ember.xcs.engine.xcs_engine import XCSPlan, XCSPlanTask

    tasks: Dict[str, XCSPlanTask] = {}
    for node_id, node in tracer_graph.nodes.items():
        tasks[node_id] = XCSPlanTask(
            node_id=node_id,
            operator=node.operator,
            inbound_nodes=node.inputs,
        )
    return XCSPlan(tasks=tasks, original_graph=tracer_graph)


def get_current_trace_context() -> Optional[TracerContext]:
    """
    Retrieve the current trace context if available.

    This stub can later be extended (for example, using thread-local storage)
    to provide global access to the current trace context.
    """
    return None
