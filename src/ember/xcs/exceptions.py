"""
Exception hierarchy for the XCS module.

This module defines a structured hierarchy of exceptions for the XCS system, enabling
more precise error handling and better diagnostics.

NOTE: This module re-exports exception classes from ember.core.exceptions
with API compatibility for backward compatibility.
"""

import logging
from typing import Any, Dict, Optional

from ember.core.exceptions import CompilationError as CoreCompilationError
from ember.core.exceptions import DataFlowError as CoreDataFlowError
from ember.core.exceptions import ExecutionError as CoreExecutionError
from ember.core.exceptions import ParallelExecutionError as CoreParallelExecutionError
from ember.core.exceptions import SchedulerError as CoreSchedulerError
from ember.core.exceptions import TraceError as CoreTraceError
from ember.core.exceptions import TransformError as CoreTransformError
from ember.core.exceptions import XCSError as CoreXCSError


# API Compatibility wrapper
class XCSError(CoreXCSError):
    """Base class for all XCS-related exceptions."""

    def __init__(self, message: str = "An error occurred in the XCS system"):
        super().__init__(message=message)
        # For backward compatibility - diagnostic_context is now self.context
        self.diagnostic_context = self.context

    def add_context(self, **kwargs: Any) -> None:
        """Adding diagnostic context to the exception.

        Storing additional metadata with the exception for improved
        traceability and debugging.

        Args:
            **kwargs: Key-value pairs to add to the diagnostic context.
        """
        super().add_context(**kwargs)
        # Keep diagnostic_context in sync with self.context
        self.diagnostic_context = self.context

    def get_context_data(self) -> Dict[str, Any]:
        """Retrieving the diagnostic context data.

        Returns:
            Dictionary containing all diagnostic context for this exception.
        """
        return self.get_context()


class TraceError(CoreTraceError, XCSError):
    """Raised when an error occurs during tracing operations."""

    def __init__(
        self,
        message: str = "Error during execution tracing",
        operation_id: Optional[str] = None,
    ):
        super().__init__(message=message)
        if operation_id:
            self.add_context(operation_id=operation_id)


class CompilationError(CoreCompilationError, XCSError):
    """Raised when an error occurs during graph compilation."""

    def __init__(
        self,
        message: str = "Error during graph compilation",
        graph_id: Optional[str] = None,
    ):
        super().__init__(message=message)
        if graph_id:
            self.add_context(graph_id=graph_id)


class ExecutionError(CoreExecutionError, XCSError):
    """Raised when an error occurs during graph execution."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        message: str = "Error during graph execution",
        cause: Optional[Exception] = None,
        **context_data: Any,
    ):
        self.node_id = node_id
        node_msg = f" in node '{node_id}'" if node_id else ""
        full_message = f"{message}{node_msg}"

        super().__init__(message=full_message, cause=cause)

        # Add standard diagnostic context
        if node_id:
            self.add_context(node_id=node_id)
        # Add any additional context provided
        if context_data:
            self.add_context(**context_data)


class TransformError(CoreTransformError, XCSError):
    """Raised when an error occurs with XCS transforms."""

    def __init__(
        self,
        transform_name: Optional[str] = None,
        message: str = "Error in XCS transform",
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
        **context_data: Any,
    ):
        self.transform_name = transform_name
        transform_msg = f" in transform '{transform_name}'" if transform_name else ""
        full_message = f"{message}{transform_msg}"

        super().__init__(message=full_message, cause=cause)

        # Add standard diagnostic context
        if transform_name:
            self.add_context(transform_name=transform_name)
        if details:
            self.add_context(**details)
        # Add any additional context provided
        if context_data:
            self.add_context(**context_data)

    def log_with_context(
        self, logger: logging.Logger, level: int = logging.ERROR
    ) -> None:
        """Logging the error with its full diagnostic context.

        Creating a structured log entry that includes all diagnostic context
        for enhanced error tracing and analysis.

        Args:
            logger: Logger to use for recording the error
            level: Logging level (default: ERROR)
        """
        super().log_with_context(logger, level)


class ParallelExecutionError(CoreParallelExecutionError, ExecutionError):
    """Raised when parallel execution fails."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        message: str = "Error during parallel execution",
        cause: Optional[Exception] = None,
        worker_id: Optional[str] = None,
        **context_data: Any,
    ):
        # Add worker-specific context for parallel execution errors
        super_context = dict(context_data)
        if worker_id:
            super_context["worker_id"] = worker_id

        super().__init__(node_id, message, cause, **super_context)


class DataFlowError(CoreDataFlowError, XCSError):
    """Raised when there is an error in data flow analysis or processing."""

    def __init__(
        self,
        message: str = "Error in data flow",
        graph_id: Optional[str] = None,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
    ):
        super().__init__(message=message)

        # Add data flow specific context
        context = {}
        if graph_id:
            context["graph_id"] = graph_id
        if source_node:
            context["source_node"] = source_node
        if target_node:
            context["target_node"] = target_node

        if context:
            self.add_context(**context)


class SchedulerError(CoreSchedulerError, XCSError):
    """Raised when there is an error in the XCS execution scheduler."""

    def __init__(
        self,
        message: str = "Error in XCS scheduler",
        graph_id: Optional[str] = None,
        scheduler_type: Optional[str] = None,
    ):
        super().__init__(message=message)

        # Add scheduler specific context
        context = {}
        if graph_id:
            context["graph_id"] = graph_id
        if scheduler_type:
            context["scheduler_type"] = scheduler_type

        if context:
            self.add_context(**context)


__all__ = [
    "XCSError",
    "TraceError",
    "CompilationError",
    "ExecutionError",
    "TransformError",
    "ParallelExecutionError",
    "DataFlowError",
    "SchedulerError",
]
