"""
XCS Tracing: Context and Metadata Types

This module defines the foundational type system for XCS execution tracing.
It provides strongly-typed data structures for capturing, transmitting, and
analyzing trace information throughout the XCS execution pipeline.

The type system is designed for:
1. Structural clarity - Using TypedDict for clear, self-documenting schemas
2. Type safety - Enabling static type checking and IDE assistance
3. Extensibility - Supporting custom metadata through standardized extension points
4. Interoperability - Facilitating integration with monitoring and debugging tools

These types form a critical part of the XCS observability infrastructure,
enabling advanced capabilities like execution profiling, debugging, audit logging,
and performance optimization while maintaining strict type safety.
"""

from typing import Dict, TypeVar, Generic
from typing_extensions import TypedDict, NotRequired


class TraceMetadata(TypedDict, total=False):
    """
    Strongly-typed schema for execution trace metadata.
    
    TraceMetadata provides a standardized, extensible structure for capturing
    contextual information about execution traces. It serves as both a data
    container and a contract for what metadata can be collected and analyzed
    during execution tracing.
    
    The schema balances comprehensiveness with flexibility by:
    - Defining common fields for universal tracing concerns (location, timing, resource usage)
    - Making all fields optional to accommodate different tracing scenarios
    - Providing an explicit extension mechanism through custom_attributes
    
    This design follows the Interface Segregation Principle by separating the
    metadata schema from trace execution and collection logic, enabling independent
    evolution of both aspects of the system.
    
    Attributes:
        source_file: Absolute path to the source file where the trace originated.
                   Useful for connecting runtime behavior back to source code.
        source_line: Line number within source_file pinpointing the exact code location.
        trace_id: Globally unique identifier for this specific trace instance.
                Used for correlation in distributed tracing scenarios.
        parent_trace_id: Reference to a parent trace when traces form a hierarchy.
                       Critical for constructing execution trees from flat trace collections.
        timestamp: Precise creation time of the trace (Unix timestamp with microsecond precision).
                 Enables accurate temporal ordering and latency calculations.
        execution_time: Total duration of the traced operation in seconds.
                      Primary metric for performance analysis and optimization.
        memory_usage: Peak memory consumption of the traced operation in bytes.
                    Used for resource utilization profiling and memory leak detection.
        custom_attributes: Extensible dictionary for domain-specific or experimental metadata.
                         Allows for tracing system extension without schema changes.
    """
    source_file: NotRequired[str]  # Path to the file where this trace was generated
    source_line: NotRequired[int]  # Line number in the source file
    trace_id: NotRequired[str]     # Unique identifier for this trace
    parent_trace_id: NotRequired[str]  # Trace ID of the parent trace if nested
    timestamp: NotRequired[float]  # When this trace was created (Unix timestamp)
    execution_time: NotRequired[float]  # Duration in seconds
    memory_usage: NotRequired[int]  # Peak memory usage in bytes
    custom_attributes: NotRequired[Dict[str, object]]  # Extension point for additional metadata


T = TypeVar('T', bound=TraceMetadata)


class TraceContextData(Generic[T]):
    """
    Generic container for trace context data with type guarantees.

    TraceContextData serves as a strongly-typed wrapper for trace metadata,
    ensuring type safety throughout the tracing subsystem. It enforces schema
    compliance while providing an extensible foundation for specialized tracing
    implementations.
    
    The class is intentionally minimal, adhering to the Single Responsibility Principle
    by focusing solely on maintaining type-safe access to trace metadata. This design
    enables new tracing capabilities to be implemented by extending this class
    without modifying the core tracing infrastructure.
    
    Key design features:
    - Generic typing with bounds ensures type safety across the tracing system
    - Immutable structure prevents metadata corruption during trace propagation
    - Simple interface minimizes implementation burden for extending systems
    - Explicit structure facilitates static analysis and IDE assistance
    
    This class forms the foundation of XCS's extensible tracing system, enabling
    specialized tracing implementations for different domains (performance profiling,
    debugging, auditing, etc.) while maintaining a unified type system.
    
    Args:
        extra_info: Strongly-typed metadata dictionary conforming to the TraceMetadata
                  schema or a schema that extends it. The generic type parameter T
                  ensures that extended implementations enforce appropriate schema typing.
    """

    def __init__(self, extra_info: T) -> None:
        self.extra_info = extra_info
