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
    """

    operator_name: str
    node_id: str
    inputs: Dict[str, Any]
    outputs: Any
    timestamp: float = field(default_factory=time.time)
    graph_node_id: Optional[str] = None


class TracerContext(ContextDecorator):
    """Context manager for capturing operator execution traces.

    When active, operator invocations may record their execution details to the active
    context. The active context is stored in thread-local storage to support safe concurrent use.

    Attributes:
        records (List[TraceRecord]): List of recorded operator invocation traces.
    """

    _local = threading.local()

    def __init__(self) -> None:
        """Initializes a new TracerContext with an empty trace record list."""
        self.records: List[TraceRecord] = []

    def __enter__(self) -> TracerContext:
        """Enters the tracing context, setting it as the current active context.

        Returns:
            TracerContext: The active tracing context.
        """
        self._set_current(self)
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
        self._clear_current()
        return None

    def add_record(self, *, record: TraceRecord) -> None:
        """Adds a trace record to the current context.

        Args:
            record (TraceRecord): The trace record to add.
        """
        self.records.append(record)

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
