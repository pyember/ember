"""XCS-specific exception types.

This module centralizes errors raised by the XCS runtime so callers can catch
and handle them explicitly.
"""

from typing import Optional


class XCSError(RuntimeError):
    """Base exception for all XCS-related failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class XCSExecutionError(XCSError):
    """Exception raised during graph execution with node context.

    Provides structured information about which node failed and the underlying
    cause, enabling better debugging of graph execution failures.

    Attributes:
        node_id: Identifier of the node that failed.
        cause: The underlying exception that triggered this error.
    """

    def __init__(
        self,
        node_id: str,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        self.node_id = node_id
        self.cause = cause
        full_message = f"Node '{node_id}': {message}"
        if cause:
            full_message = f"{full_message} (caused by {type(cause).__name__}: {cause})"
        super().__init__(full_message)


__all__ = ["XCSError", "XCSExecutionError"]
