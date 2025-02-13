"""
JIT Decorator for XCS Operators.

This module instruments an Operator subclass so that on its first invocation the operator
execution is traced to produce an execution plan (via the unified TracerContext from
xcs_tracing.py). All subsequent invocations run the pre-compiled plan.
"""

import functools
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from ember.xcs.tracer.xcs_tracing import TracerContext, convert_traced_graph_to_plan
from ember.xcs.engine.xcs_engine import execute_graph
from ember.core.registry.operator.core.operator_base import Operator

T = TypeVar("T", bound=Operator)


def jit(sample_input: Optional[Dict[str, Any]] = None) -> Callable[[Type[T]], Type[T]]:
    """Decorator that augments an Operator subclass with JIT tracing capabilities.

    When applied, the decorated Operator subclass will trace its execution on first use
    to generate and cache an execution plan. If a sample input is provided during initialization,
    the plan is compiled eagerly; otherwise, lazy tracing is performed upon the first call.

    Args:
        sample_input (Optional[Dict[str, Any]]): A dictionary used as a sample of the input for
            eager plan compilation. If None, tracing and plan compilation are deferred until needed.

    Returns:
        Callable[[Type[T]], Type[T]]: A class decorator that injects JIT tracing and plan
            compilation into the Operator subclass.

    Raises:
        TypeError: If the decorator is applied to a class that is not a subclass of Operator.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, Operator):
            raise TypeError("@jit can only be applied to an Operator subclass.")

        original_init: Callable[..., None] = cls.__init__
        original_call: Callable[..., Any] = cls.__call__

        @functools.wraps(original_init)
        def new_init(self: T, *args: Any, **kwargs: Any) -> None:
            """Initializes the operator instance with attributes for JIT tracing.

            This modified initializer calls the original __init__ and then sets up instance
            attributes necessary for tracing and plan caching. If a sample input is provided,
            tracing and plan compilation occur immediately.

            Args:
                *args: Positional arguments for the original initializer.
                **kwargs: Keyword arguments for the original initializer.
            """
            original_init(self, *args, **kwargs)
            self._compiled_plan = None
            self._jit_traced = False
            self._in_tracing = False
            if sample_input is not None:
                self._trace_and_compile(sample_input=sample_input)

        def jit_call(self: T, inputs: Any) -> Any:
            """Executes the operator using a pre-compiled execution plan if available.

            On the first call, if tracing has not yet been performed, it validates the inputs,
            initiates tracing to compile the execution plan, and marks the operator as traced.
            On subsequent calls, or if a plan was compiled eagerly, the cached execution plan is used.

            Args:
                inputs (Any): The input data to the operator.

            Returns:
                Any: The output produced by executing the operator, either by running the cached plan
                    or by directly invoking the original __call__ method.

            Raises:
                Exception: Propagates any exception raised during operator execution.
            """
            validated_inputs: Any = self.get_signature().validate_inputs(inputs)
            if not self._jit_traced:
                self._trace_and_compile(sample_input=validated_inputs)
                self._jit_traced = True
            if self._compiled_plan is not None:
                return execute_graph(
                    graph=self._compiled_plan, global_input=validated_inputs
                )
            raw_output: Any = original_call(self, validated_inputs)
            return self.get_signature().validate_output(raw_output)

        def _trace_and_compile(self: T, sample_input: Any) -> None:
            """Traces the operator to build and cache an execution plan.

            This method safeguards against re-entrant tracing. It uses the provided sample input
            to execute a trace within a TracerContext, then compiles and caches the traced graph into
            an executable plan.

            Args:
                sample_input (Any): The sample input used for tracing the operator execution.

            Returns:
                None
            """
            if self._jit_traced or getattr(self, "_in_tracing", False):
                return
            self._in_tracing = True
            try:
                with TracerContext(
                    top_operator=self, sample_input=sample_input
                ) as tctx:
                    tracer_graph: Any = tctx.run_trace()
                plan: Any = convert_traced_graph_to_plan(tracer_graph=tracer_graph)
                self._compiled_plan = plan
                self._jit_traced = True
            finally:
                self._in_tracing = False

        setattr(cls, "__init__", new_init)
        setattr(cls, "__call__", jit_call)
        setattr(cls, "_trace_and_compile", _trace_and_compile)
        return cls

    return decorator
