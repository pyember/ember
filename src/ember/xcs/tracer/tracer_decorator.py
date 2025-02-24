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
import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.xcs.tracer.xcs_tracing import TraceRecord, TracerContext

# Type variable for Operator subclasses.
T = TypeVar("T", bound=Operator)

def jit(
    *,
    sample_input: Optional[Dict[str, Any]] = None,
    force_trace: bool = False,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator that instruments an Operator for execution tracing.

    When applied, the operator's __call__ method is wrapped so that each invocation
    records a trace record (if a TracerContext is active or if forced). The trace includes
    the operator's name, a unique node identifier, input parameters, output result, and a timestamp.

    Args:
        sample_input (Optional[Dict[str, Any]]): Sample input for tracing purposes.
            (Not used in this simplified version.)
        force_trace (bool): If True, forces tracing on every call regardless of caching.
            Defaults to False.

    Returns:
        Callable[[Type[T]], Type[T]]: The decorated Operator subclass.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, Operator):
            raise TypeError("@jit_trace decorator can only be applied to an Operator subclass.")

        original_call = cls.__call__

        @functools.wraps(original_call)
        def traced_call(self: T, *, inputs: Dict[str, Any]) -> Any:
            """
            Wrapped __call__ method that records execution trace.

            Args:
                inputs (Dict[str, Any]): The input parameters for the operator.

            Returns:
                Any: The output from the operator execution.
            """
            tracer: Optional[TracerContext] = TracerContext.get_current()
            start_time = time.time()
            output = original_call(self, inputs=inputs)
            end_time = time.time()

            if tracer is not None or force_trace:
                record = TraceRecord(
                    operator_name=getattr(self, "name", self.__class__.__name__),
                    node_id=str(id(self)),
                    inputs=inputs,
                    outputs=output,
                    timestamp=end_time,
                )
                if tracer is not None:
                    tracer.add_record(record=record)
            return output

        cls.__call__ = traced_call  # type: ignore
        return cls

    return decorator