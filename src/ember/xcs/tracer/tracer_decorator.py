"""
Tracer Decorator for XCS Operators.

This module provides a decorator that instruments an Operator
subclass for execution tracing. Upon first invocation, the operator's execution is traced
symbolically. The tracer leverages PyTree flattening (via EmberModule) and
records operations into an IR graph (consisting of IRNode objects). Subsequent
calls execute the cached plan.

Allows optional forced tracing on every call and customizable caching logic.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.xcs.tracer.xcs_tracing import TraceRecord, TracerContext

# Type variable for Operator subclasses.
OperatorType = TypeVar("OperatorType", bound=Operator)
# Type alias for the decorator function's return type
OperatorDecorator = Callable[[Type[OperatorType]], Type[OperatorType]]


def jit(
    *,
    sample_input: Optional[Dict[str, Any]] = None,
    force_trace: bool = False,
) -> OperatorDecorator:
    """Decorator that instruments an Operator for execution tracing.

    When applied, the operator's __call__ method is wrapped to record execution traces
    when a TracerContext is active or if tracing is forced. The trace includes
    the operator's name, node identifier, inputs, outputs, and timing information.

    Args:
        sample_input (Optional[Dict[str, Any]]): Optional pre-defined input for compilation/tracing.
            Not used yet in the current implementation but reserved for future use.
        force_trace: If True, traces every invocation regardless of caching.
            Defaults to False.

    Returns:
        A decorator function that transforms the Operator subclass.

    Raises:
        TypeError: If applied to a class that doesn't inherit from Operator.
    """

    def decorator(cls: Type[OperatorType]) -> Type[OperatorType]:
        """Internal decorator function applied to the Operator class.

        Args:
            cls: The Operator subclass to be instrumented.

        Returns:
            The decorated Operator class with tracing capabilities.

        Raises:
            TypeError: If cls is not an Operator subclass.
        """
        if not issubclass(cls, Operator):
            raise TypeError(
                "@jit decorator can only be applied to an Operator subclass."
            )

        original_call = cls.__call__

        @functools.wraps(original_call)
        def traced_call(self: OperatorType, *, inputs: Dict[str, Any]) -> Any:
            """Wrapped __call__ method that records execution trace.

            Args:
                inputs (Dict[str, Any]): The input parameters for the operator.

            Returns:
                Any: The output from the operator execution.
            """
            tracer: Optional[TracerContext] = TracerContext.get_current()
            start_time = time.time()
            output = original_call(self=self, inputs=inputs)
            end_time = time.time()

            if tracer is not None or force_trace:
                # Get operator name, preferring the 'name' attribute if available
                operator_name = getattr(self, "name", self.__class__.__name__)

                record = TraceRecord(
                    operator_name=operator_name,
                    node_id=str(id(self)),
                    inputs=inputs,
                    outputs=output,
                    timestamp=end_time,
                )

                if tracer is not None:
                    tracer.add_record(record=record)

            return output

        # Replace the original __call__ method with our traced version
        cls.__call__ = cast(Callable, traced_call)
        return cls

    return decorator
