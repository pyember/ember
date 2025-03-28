"""Types for XCS execution tracing.

Defines the type system for tracking and propagating trace information through
the XCS execution pipeline. Provides structured metadata for debugging, profiling,
and analysis.
"""

from typing import Dict, Generic, TypeVar

from typing_extensions import NotRequired, TypedDict


class TraceMetadata(TypedDict, total=False):
    """Schema for execution trace metadata.

    Attributes:
        source_file: Path to the source file where the trace originated
        source_line: Line number within source_file
        trace_id: Unique identifier for this trace instance
        parent_trace_id: Reference to parent trace for hierarchical tracing
        timestamp: Creation time (Unix timestamp)
        execution_time: Duration in seconds
        memory_usage: Peak memory usage in bytes
        custom_attributes: Dictionary for domain-specific metadata
    """

    source_file: NotRequired[str]
    source_line: NotRequired[int]
    trace_id: NotRequired[str]
    parent_trace_id: NotRequired[str]
    timestamp: NotRequired[float]
    execution_time: NotRequired[float]
    memory_usage: NotRequired[int]
    custom_attributes: NotRequired[Dict[str, object]]


T = TypeVar("T", bound=TraceMetadata)


class TraceContextData(Generic[T]):
    """Container for trace context data with type guarantees.

    Args:
        extra_info: Metadata dictionary conforming to TraceMetadata schema
    """

    def __init__(self, extra_info: T) -> None:
        self.extra_info = extra_info
