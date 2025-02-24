"""
Tracer Decorator for XCS Operators.

This module provides a decorator that instruments an Operator
subclass so that on its first invocation the operator's execution is traced
symbolically. The tracer leverages PyTree flattening (via EmberModule) and
records operations into an IR graph (consisting of IRNode objects). Subsequent
calls execute the cached plan.

Allows optional forced tracing on every call and customizable caching logic.
"""

from __future__ import annotations

import functools
import threading
import uuid
import logging
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

# Import necessary modules from our codebase.
from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.xcs.graph.xcs_graph import XCSGraph
from src.ember.xcs.engine.xcs_engine import execute_graph, XCSPlan, compile_graph
from src.ember.xcs.utils.tree_util import tree_flatten, tree_unflatten

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Type variable for Operators.
T = TypeVar("T", bound=Operator)


# ------------------------------------------------------------------------------
# IR Graph and Node Definitions
# ------------------------------------------------------------------------------


@dataclass
class IRNode:
    """
    Represents a node in the intermediate representation (IR) graph.
    For inputs, operator can be None.
    """

    node_id: str
    operator: Optional[Operator]
    inputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRGraph:
    """
    Represents an IR graph constructed from tracing an operator.
    """

    nodes: Dict[str, IRNode] = field(default_factory=dict)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)

    def add_node(
        self,
        *,
        operator: Optional[Operator],
        inputs: List[str],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> str:
        node_id: str = str(uuid.uuid4())[:8]
        self.nodes[node_id] = IRNode(
            node_id=node_id,
            operator=operator,
            inputs=inputs,
            attrs=attrs or {},
        )
        return node_id


# ------------------------------------------------------------------------------
# Tracer Context using PyTree Flattening
# ------------------------------------------------------------------------------


class TracerContext(ContextDecorator):
    """
    Tracer context to capture a full operator call and construct an IR graph
    based on PyTree flattening of inputs and outputs.
    """

    # Use thread-local storage for current tracer context.
    _local = threading.local()

    def __init__(self, *, top_operator: Operator, sample_input: Dict[str, Any]) -> None:
        self.top_operator = top_operator
        self.sample_input: Dict[str, Any] = sample_input
        self.ir_graph = IRGraph()
        self._input_tracers: Dict[str, str] = {}

    def __enter__(self) -> TracerContext:
        self._set_current(self)
        # Flatten the sample_input so each leaf can become a node
        leaves, aux = tree_flatten(tree=self.sample_input)
        for i, leaf in enumerate(leaves):
            node_id = self.ir_graph.add_node(
                operator=None,
                inputs=[],
                attrs={"value": leaf, "index": i},
            )
            key = f"input_{i}"
            self.ir_graph.input_mapping[key] = node_id
            self._input_tracers[key] = node_id
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> Optional[bool]:
        self._clear_current()
        return None

    @classmethod
    def get_current(cls) -> Optional[TracerContext]:
        return getattr(cls._local, "current", None)

    def _set_current(self, ctx: TracerContext) -> None:
        cls = type(self)
        cls._local.current = ctx

    def _clear_current(self) -> None:
        cls = type(self)
        cls._local.current = None

    def run_trace(self) -> IRGraph:
        # Reconstruct inputs
        leaves, aux = tree_flatten(tree=self.sample_input)
        unflattened_inputs = tree_unflatten(aux=aux, children=leaves)

        # Add the operator node
        operator_node_id = self.ir_graph.add_node(
            operator=self.top_operator,
            inputs=list(self._input_tracers.values()),
        )

        # Invoke the operator with reconstructed inputs
        output = self.top_operator(inputs=cast(Dict[str, Any], unflattened_inputs))

        # Flatten and record the output leaves
        out_leaves, out_aux = tree_flatten(tree=output)
        for i, leaf in enumerate(out_leaves):
            self.ir_graph.output_mapping[f"output_{i}"] = operator_node_id
            # Store the output leaves in the operator node for reference
            self.ir_graph.nodes[operator_node_id].attrs[f"output_{i}"] = leaf

        _LOGGER.debug("Constructed IR graph with %d nodes.", len(self.ir_graph.nodes))
        return self.ir_graph


# ------------------------------------------------------------------------------
# JIT Decorator for Operator Tracing/Compilation
# ------------------------------------------------------------------------------


def jit(
    *,
    sample_input: Optional[Dict[str, Any]] = None,
    force_trace_forward: bool = False,
    cache_key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    JIT decorator that applies a single trace of an operator and compiles the execution
    plan. Subsequent calls use the cached plan unless force_trace_forward is True or
    the cache key is different.

    Args:
        sample_input: Optional sample input for forcing a known shape or structure.
        force_trace_forward: If True, trace on every call rather than caching.
        cache_key_fn: A function that computes a string key from inputs for plan caching.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, Operator):
            raise TypeError(
                "@jit decorator can only be applied to an Operator subclass."
            )

        original_init = cls.__init__
        original_call = cls.__call__

        @functools.wraps(original_init)
        def new_init(self: T, *args: Any, **kwargs: Any) -> None:
            # Call the original constructor
            original_init(self, *args, **kwargs)

            # Store the original call method for raw invocations
            self._jit_original_call = Operator.__call__.__get__(self, cls)

            # Dictionary for compiled plans keyed by a string (default or user-provided)
            self._compiled_plans: Dict[str, XCSPlan] = {}

            # Track which cache keys have traced successfully
            self._jit_traced: Dict[str, bool] = {}

            # Thread-local boolean to indicate if we are currently tracing
            self._in_tracing = threading.local()
            self._in_tracing.value = False

            # Thread-local trace depth
            self._trace_depth = threading.local()
            self._trace_depth.value = 0

            # Save config
            self._force_trace_forward = force_trace_forward
            if sample_input is None:
                # With no sample_input, use a constant key ("default") to reuse the same compiled plan.
                self._cache_key_fn = cache_key_fn or (lambda inps: "default")
            else:
                self._cache_key_fn = cache_key_fn or (lambda inps: str(sorted(inps.items())))

        @functools.wraps(original_call)
        def new_call(self: T, *, inputs: Dict[str, Any]) -> Any:
            validated_inputs = self.signature.validate_inputs(inputs=inputs)
            cache_key = self._cache_key_fn(validated_inputs)

            # If we are inside a tracing call, just do the raw operator call
            if getattr(self._trace_depth, "value", 0) > 0:
                return self._jit_original_call(inputs=validated_inputs)

            # If forced or not yet traced for this key, do a trace
            if self._force_trace_forward or cache_key not in self._jit_traced:
                self._trace_and_compile(
                    trace_input=validated_inputs, cache_key=cache_key
                )
                self._jit_traced[cache_key] = True

            # Execute the compiled plan with the current validated inputs,
            # caching the result only if the new inputs match the ones used previously.
            plan = self._compiled_plans[cache_key]
            if (
                (not self._force_trace_forward)
                and hasattr(plan, "_cached_input")
                and plan._cached_input == validated_inputs
            ):
                result = plan._cached_output
            else:
                result = execute_graph(
                    graph=plan,
                    global_input=validated_inputs,
                    concurrency=True,
                )
                if not self._force_trace_forward:
                    plan._cached_input = validated_inputs
                    plan._cached_output = result

            return result if "compiled_root" not in result else result["compiled_root"]

        def _trace_and_compile(
            self: T, *, trace_input: Dict[str, Any], cache_key: str
        ) -> None:
            """
            Performs the actual tracing by creating a TracerContext, capturing the IR,
            and compiling it to an XCSPlan. The resulting plan is stored in self._compiled_plans[cache_key].
            """
            if getattr(self._in_tracing, "value", False):
                return

            self._trace_depth.value += 1
            self._in_tracing.value = True
            original_call_method = self.__call__
            try:
                # Temporarily override __call__ with raw call to avoid recursion
                self.__call__ = self._jit_original_call

                # Use the provided sample_input if specified, otherwise use the current trace_input
                sample = trace_input if sample_input is None else sample_input
                with TracerContext(top_operator=self, sample_input=sample) as tracer:
                    ir_graph = tracer.run_trace()
                plan = compile_graph(
                    graph=_convert_ir_graph_to_xcs_graph(
                        ir_graph=ir_graph, operator=self
                    )
                )
                self._compiled_plans[cache_key] = plan
            finally:
                self.__call__ = original_call_method
                self._in_tracing.value = False
                self._trace_depth.value -= 1

        cls.__init__ = new_init
        cls.__call__ = new_call
        setattr(cls, "_trace_and_compile", _trace_and_compile)
        return cls

    return decorator


# ------------------------------------------------------------------------------
# Conversion Helper: IRGraph to XCSGraph
# ------------------------------------------------------------------------------


def _convert_ir_graph_to_xcs_graph(
    *, ir_graph: IRGraph, operator: Operator
) -> XCSGraph:
    """
    Converts an IRGraph (obtained via tracing) into an XCSGraph that wraps the operator call.
    For simplicity, all input nodes feed into a single 'compiled_root' node that calls
    the operator with validated inputs.
    """
    xcs_graph: XCSGraph = XCSGraph()

    def compiled_operator(*, inputs: Dict[str, Any]) -> Any:
        validated_inputs = operator.signature.validate_inputs(inputs=inputs)
        return operator.forward(inputs=validated_inputs)

    # Add a root with the traced operator
    root_id: str = xcs_graph.add_node(
        operator=compiled_operator, node_id="compiled_root"
    )

    # Only process nodes that represent inputs. In our IRGraph,
    # input nodes are created with operator=None and have a "value" attribute.
    for node_id, node in ir_graph.nodes.items():
        if node.operator is None:
            if "value" in node.attrs:
                # Wrap the primitive value in a callable lambda.
                xcs_graph.add_node(
                    operator=lambda *, inputs, _val=node.attrs["value"]: _val,
                    node_id=node_id,
                )
                xcs_graph.add_edge(from_id=node_id, to_id=root_id)
            else:
                # Warn if an input node is missing its expected "value".
                _LOGGER.warning("IR node %s is missing a 'value' attribute; skipping.", node_id)
        else:
            # In this simplified conversion, we expect exactly one node with an operator,
            # which is used as the compiled root. Other nodes with a callable operator are ignored.
            _LOGGER.debug("Skipping non-input IR node: %s", node_id)

    return xcs_graph
