"""
This module provides classes for recording operator trace logs using a thread-safe context variable.
"""

from __future__ import annotations

import contextvars
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

# Thread-local ContextVar that holds the active TraceContext, if any.
_CURRENT_TRACE_CONTEXT: contextvars.ContextVar[Optional[TraceContext]] = (
    contextvars.ContextVar("current_trace_context", default=None)
)


class TraceContext:
    """Context for recording operator trace logs.

    This context maintains a collection of TraceRecord instances,
    each encapsulating details of an operator invocation, including its
    inputs, outputs, and any optional custom metadata.

    Attributes:
        records (List[TraceRecord]): A list of recorded operator invocation logs.
    """

    def __init__(self) -> None:
        """Initialize a new TraceContext with an empty record list."""
        self.records: List[TraceRecord] = []

    def add_record(self, *, record: TraceRecord) -> None:
        """Add a trace record to the context.

        Args:
            record (TraceRecord): The trace record describing an operator invocation.
        """
        self.records.append(record)

    def get_records(self) -> List[TraceRecord]:
        """Retrieve all trace records from the context.

        Returns:
            List[TraceRecord]: The list of all recorded operator trace logs.
        """
        return self.records


class TraceRecord:
    """Encapsulates the record of a single operator invocation.

    Attributes:
        operator_name (str): The identifier name of the operator.
        operator_class (str): The class name of the operator.
        input_data (Any): The raw input data provided to the operator.
        output_data (Any): The raw output data produced by the operator.
        metadata (Dict[str, Any]): Custom metadata related to the invocation.
    """

    def __init__(
        self,
        *,
        operator_name: str,
        operator_class: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.operator_name: str = operator_name
        self.operator_class: str = operator_class
        self.input_data: Any = input_data
        self.output_data: Any = output_data
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}


class RecordingContext:
    """Context manager for establishing a TraceContext for operator logging.

    Upon entering the context, a new TraceContext is created and set as
    the active context. Operator invocations within this scope can record
    their execution details to this active context. Exiting the context
    restores the previous TraceContext.
    """

    def __enter__(self) -> TraceContext:
        """Enter the recording context by setting up a new TraceContext.

        Returns:
            TraceContext: The newly created TraceContext instance that becomes active.
        """
        new_context: TraceContext = TraceContext()
        self._context_token: contextvars.Token = _CURRENT_TRACE_CONTEXT.set(new_context)
        return new_context

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit the recording context, restoring the previous TraceContext.

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type if an exception occurred; otherwise, None.
            exc_value (Optional[BaseException]): The exception instance if an exception occurred; otherwise, None.
            traceback (Optional[TracebackType]): The traceback if an exception occurred; otherwise, None.
        """
        _CURRENT_TRACE_CONTEXT.reset(self._context_token)


def get_current_trace_context() -> Optional[TraceContext]:
    """Retrieve the currently active TraceContext.

    Returns:
        Optional[TraceContext]: The active TraceContext, or None if there is no active context.
    """
    return _CURRENT_TRACE_CONTEXT.get()
