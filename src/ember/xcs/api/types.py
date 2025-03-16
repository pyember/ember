"""
Type definitions for the XCS API.

This module provides type definitions for the XCS API, ensuring type safety and
proper interface contracts. These types are used throughout the XCS system to
maintain consistency and clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.xcs_tracing import TraceRecord

# Type variables for generic operators
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


@dataclass
class XCSExecutionOptions:
    """Configuration options for XCS execution."""

    max_workers: int = 10
    """Maximum number of concurrent workers for parallel execution."""

    timeout: Optional[float] = None
    """Optional timeout in seconds for execution."""

    cache_results: bool = True
    """Whether to cache execution results."""

    debug_mode: bool = False
    """Whether to enable debug mode with additional logging."""


@dataclass
class ExecutionResult:
    """Result of executing a graph or operation."""

    outputs: Dict[str, Any]
    """The outputs from the execution."""

    execution_time: float
    """Time taken for execution in seconds."""

    node_stats: Optional[Dict[str, Dict[str, Any]]] = None
    """Optional statistics about individual node execution."""


@dataclass
class JITOptions:
    """Configuration options for JIT compilation."""

    sample_input: Optional[Dict[str, Any]] = None
    """Sample input for eager compilation."""

    force_trace: bool = False
    """Whether to force tracing on every execution."""

    recursive: bool = True
    """Whether to trace recursively."""

    cache_key_fn: Optional[Callable[[Dict[str, Any]], str]] = None
    """Optional function to create cache keys from inputs."""


@dataclass
class TransformOptions:
    """Configuration options for transforms like vmap and pmap."""

    in_axes: Optional[Union[int, Dict[str, int]]] = 0
    """Input axes for vectorization/parallelization."""

    out_axes: Optional[int] = 0
    """Output axes for vectorization/parallelization."""

    devices: Optional[List[Any]] = None
    """Devices for parallelization."""


@runtime_checkable
class GraphBuilder(Protocol):
    """Protocol for graph builders."""

    def build_graph(self, records: List[TraceRecord]) -> XCSGraph:
        """
        Build a graph from trace records.

        Args:
            records: List of trace records

        Returns:
            An XCS graph for execution
        """
        ...
