"""
Tracing Module for XCS.

This module provides a context manager for tracing operator executions and recording
trace records.
"""

from __future__ import annotations

import threading
import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass(frozen=False)  # Allow modifications to instances
class TraceRecord:
    """Trace record for an operator invocation.

    Attributes:
        operator_name (str): Name of the operator.
        node_id (str): Unique identifier for the operator instance.
        inputs (Dict[str, Any]): The inputs passed to the operator.
        outputs (Any): The outputs returned by the operator.
        timestamp (float): The time at which the operator finished execution.
        graph_node_id (Optional[str]): ID used in the graph representation, for autograph internals.
        operator (Any): The operator instance that was called.
    """

    operator_name: str
    node_id: str
    inputs: Dict[str, Any]
    outputs: Any
    timestamp: float = field(default_factory=time.time)
    graph_node_id: Optional[str] = None
    operator: Any = None


class TracerContext(ContextDecorator):
    """Context manager for capturing operator execution traces.

    When active, operator invocations may record their execution details to the active
    context. The active context is stored in thread-local storage to support safe concurrent use.

    Attributes:
        records (List[TraceRecord]): List of recorded operator invocation traces.
        calls: Dictionary mapping call IDs to tracked operator calls
        is_active: Whether this context is currently active
    """

    _local = threading.local()

    def __init__(self) -> None:
        """Initializes a new TracerContext with an empty trace record list."""
        self.records: List[TraceRecord] = []
        self.is_active: bool = False
        self.calls: Dict[str, TraceRecord] = {}

    def __enter__(self) -> TracerContext:
        """Enters the tracing context, setting it as the current active context.

        Returns:
            TracerContext: The active tracing context.
        """
        self._set_current(self)
        self.is_active = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> Optional[bool]:
        """Exits the tracing context, clearing the active context.

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type, if any.
            exc_value (Optional[BaseException]): Exception value, if any.
            traceback (Optional[Any]): Traceback, if any.

        Returns:
            Optional[bool]: None.
        """
        self.is_active = False
        self._clear_current()
        return None

    def add_record(self, *, record: TraceRecord) -> None:
        """Adds a trace record to the current context.

        Args:
            record (TraceRecord): The trace record to add.
        """
        self.records.append(record)

    def track_call(self, operator: Any, inputs: Dict[str, Any]) -> str:
        """Track an operator call in the context.

        Args:
            operator: The operator being called
            inputs: The inputs to the operator

        Returns:
            A unique ID for this call
        """
        call_id = str(len(self.calls) + 1)
        record = TraceRecord(
            operator_name=operator.__class__.__name__,
            node_id=call_id,
            inputs=inputs,
            outputs=None,
            operator=operator,
        )
        self.calls[call_id] = record
        return call_id

    def get_call(self, call_id: str) -> Optional[TraceRecord]:
        """Get a tracked call by ID.

        Args:
            call_id: The call ID to look up

        Returns:
            The TraceRecord for the call, or None if not found
        """
        return self.calls.get(call_id)

    @classmethod
    def get_current(cls) -> Optional[TracerContext]:
        """Retrieves the current active tracing context.

        Returns:
            Optional[TracerContext]: The active TracerContext, or None if none is active.
        """
        return getattr(cls._local, "current", None)

    def _set_current(self, ctx: TracerContext) -> None:
        """Sets the current tracing context in thread-local storage.

        Args:
            ctx (TracerContext): The tracer context to set as current.
        """
        type(self)._local.current = ctx

    def _clear_current(self) -> None:
        """Clears the current tracing context from thread-local storage."""
        type(self)._local.current = None


def get_tracing_context() -> Optional[TracerContext]:
    """Get the current active tracing context.

    This is a helper function that simply delegates to TracerContext.get_current()
    for convenience.

    Returns:
        The current active tracing context, or None if no context is active.
    """
    # Make sure there's always a context available
    context = TracerContext.get_current()
    if context is None:
        context = TracerContext()
        context._set_current(context)
    return context


# Dictionary to store original implementations when patching
_ORIGINAL_METHODS: Dict[int, Any] = {}


def patch_operator(operator: Any, new_method: Any) -> None:
    """Replace an operator's __call__ method with a new implementation.

    This function is primarily used for testing and debugging purposes
    to intercept operator calls.

    Args:
        operator: The operator to patch
        new_method: The new __call__ method implementation
    """
    # Store the original method
    operator_id = id(operator)
    _ORIGINAL_METHODS[operator_id] = operator.__call__

    # Apply the patch
    operator.__call__ = new_method


def restore_operator(operator: Any) -> None:
    """Restore an operator's original __call__ method after patching.

    Args:
        operator: The operator to restore
    """
    operator_id = id(operator)
    if operator_id in _ORIGINAL_METHODS:
        # Restore the original method
        operator.__call__ = _ORIGINAL_METHODS[operator_id]
        # Clean up the reference
        del _ORIGINAL_METHODS[operator_id]
