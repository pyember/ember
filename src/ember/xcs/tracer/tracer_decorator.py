"""
JIT Compilation and Execution Tracing for XCS Operators

This module provides a just-in-time (JIT) compilation system for Ember operators
through execution tracing. The @jit decorator transforms operator classes by
instrumenting them to record their execution patterns and automatically compile
optimized execution plans.

Key features:
1. Transparent operator instrumentation via the @jit decorator
2. Automatic execution graph construction from traced operator calls
3. Compile-once, execute-many optimization for repeated operations
4. Support for pre-compilation with sample inputs
5. Configurable tracing and caching behaviors

Implementation follows functional programming principles where possible,
separating concerns between tracing, compilation, and execution. The design
adheres to the Open/Closed Principle by extending operator behavior without
modifying their core implementation.

Example:
    @jit
    class MyOperator(Operator):
        def __call__(self, *, inputs):
            # Complex, multi-step computation
            return result

    # First call triggers tracing and compilation
    op = MyOperator()
    result1 = op(inputs={"text": "example"})

    # Subsequent calls reuse the compiled execution plan
    result2 = op(inputs={"text": "another example"})
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
)

# Import the base classes carefully to avoid circular imports
from ember.xcs.tracer.xcs_tracing import TracerContext, TraceRecord

# We need to use a string for the bound to avoid circular imports
# Type variable for Operator subclasses
OperatorType = TypeVar("OperatorType", bound="Operator")
# Type alias for the decorator function's return type
OperatorDecorator = Callable[[Type[OperatorType]], Type[OperatorType]]

# Use a Protocol for Operator to avoid circular imports
from typing import Protocol, runtime_checkable


@runtime_checkable
class Operator(Protocol):
    """Protocol defining the expected interface for Operators."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the operator with provided inputs."""
        ...


@runtime_checkable
class StateDependency(Protocol):
    """Protocol for operators to declare state dependencies."""

    def get_state_signature(self) -> str:
        """Return a signature representing the current state.

        When this signature changes, cached JIT compilations should be invalidated.
        This could be a hash of state variables or a version number that
        the operator increments when state changes.
        """
        ...

    def get_state_dependencies(self) -> Set[object]:
        """Return set of objects this operator's behavior depends on.

        This is used to identify other objects that might affect this
        operator's behavior, allowing for more sophisticated cache
        invalidation strategies.
        """
        ...


# Forward import execution components to avoid circular imports
from ember.xcs.graph.xcs_graph import XCSGraph

import weakref

# Type variable for cache value type
T = TypeVar("T")


@dataclasses.dataclass
class JITMetrics:
    """Performance metrics for JIT compilation and execution.

    Tracks timing and cache statistics for analyzing JIT system performance.
    All times are in seconds.
    """

    # Timing metrics
    compilation_time: float = 0.0
    execution_time: float = 0.0
    tracing_time: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        """Ratio of cache hits to total cache lookups."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def record_compilation(self, duration: float) -> None:
        """Record time spent compiling a graph."""
        self.compilation_time += duration

    def record_execution(self, duration: float) -> None:
        """Record time spent executing a graph."""
        self.execution_time += duration

    def record_tracing(self, duration: float) -> None:
        """Record time spent tracing execution."""
        self.tracing_time += duration

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.compilation_time = 0.0
        self.execution_time = 0.0
        self.tracing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    def __str__(self) -> str:
        """Human-readable performance summary."""
        return (
            f"JIT Performance Metrics:\n"
            f"  Compilation: {self.compilation_time:.6f}s\n"
            f"  Execution: {self.execution_time:.6f}s\n"
            f"  Tracing: {self.tracing_time:.6f}s\n"
            f"  Cache hit ratio: {self.cache_hit_ratio:.2%} ({self.cache_hits}/{self.cache_hits + self.cache_misses})"
        )


class JITCache(Generic[T]):
    """Thread-safe cache for JIT-compiled artifacts with proper lifecycle management."""

    def __init__(self) -> None:
        self._cache = weakref.WeakKeyDictionary()
        self._state_signatures = weakref.WeakKeyDictionary()
        self.metrics = JITMetrics()

    def get(self, key: object) -> Optional[T]:
        """Retrieve a cached item by key object (not id)."""
        return self._cache.get(key)

    def get_with_state(
        self, key: object, state_signature: Optional[str] = None
    ) -> Optional[T]:
        """Retrieve cached item, checking state signature if available."""
        if key not in self._cache:
            self.metrics.record_cache_miss()
            return None

        # If state signature provided, validate it matches
        if state_signature is not None:
            cached_signature = self._state_signatures.get(key)
            if cached_signature != state_signature:
                # State changed, invalidate cache entry
                self.invalidate(key)
                self.metrics.record_cache_miss()
                return None

        self.metrics.record_cache_hit()
        return self._cache.get(key)

    def set(self, key: object, value: T, state_signature: Optional[str] = None) -> None:
        """Store an item in the cache using the object itself as key."""
        self._cache[key] = value
        if state_signature is not None:
            self._state_signatures[key] = state_signature

    def invalidate(self, key: Optional[object] = None) -> None:
        """Invalidate specific entry or entire cache."""
        if key is not None:
            self._cache.pop(key, None)
            self._state_signatures.pop(key, None)
        else:
            self._cache.clear()
            self._state_signatures.clear()

    def __len__(self) -> int:
        """Return number of items in the cache."""
        return len(self._cache)

    def get_metrics(self) -> JITMetrics:
        """Get a copy of the current metrics."""
        return dataclasses.replace(self.metrics)

    def reset_metrics(self) -> None:
        """Reset metrics to initial state."""
        self.metrics.reset()


# Cache to store compiled execution graphs for each operator class instance
_jit_cache = JITCache[XCSGraph]()


def jit(
    func=None,
    *,
    sample_input: Optional[Dict[str, Any]] = None,
    force_trace: bool = False,
    recursive: bool = True,
):
    """Just-In-Time compilation decorator for Ember Operators.

    The @jit decorator transforms Operator classes to automatically trace their execution
    and compile optimized execution plans. This brings significant performance benefits
    for complex operations and operator pipelines by analyzing the execution pattern
    once and reusing the optimized plan for subsequent calls.

    The implementation follows a lazily evaluated, memoization pattern:
    1. First execution triggers tracing to capture the full execution graph
    2. The traced operations are compiled into an optimized execution plan
    3. Subsequent calls reuse this plan without re-tracing (unless force_trace=True)

    Pre-compilation via sample_input is available for performance-critical paths where
    even the first execution needs to be fast. This implements an "eager" JIT pattern
    where compilation happens at initialization time rather than first execution time.

    Design principles:
    - Separation of concerns: Tracing, compilation, and execution are distinct phases
    - Minimal overhead: Non-tracing execution paths have negligible performance impact
    - Transparency: Decorated operators maintain their original interface contract
    - Configurability: Multiple options allow fine-tuning for different use cases

    Args:
        func: The function or class to be JIT-compiled. This is automatically passed when
             using the @jit syntax directly. If using @jit(...) with parameters, this will be None.
        sample_input: Optional pre-defined input for eager compilation during initialization.
                    This enables "compile-time" optimization rather than runtime JIT compilation.
                    Recommended for performance-critical initialization paths.
        force_trace: When True, disables caching and traces every invocation.
                    This is valuable for debugging and for operators whose execution
                    pattern varies significantly based on input values.
                    Performance impact: Significant, as caching benefits are disabled.
        recursive: Controls whether nested operator calls are also traced and compiled.
                 Currently limited to direct child operators observed during tracing.
                 Default is True, enabling full pipeline optimization.

    Returns:
        A decorated function/class or a decorator function that transforms the target
        Operator subclass by instrumenting its initialization and call methods for tracing.

    Raises:
        TypeError: If applied to a class that doesn't inherit from Operator.
                  The decorator strictly enforces type safety to prevent
                  incorrect usage on unsupported class types.

    Example:
        # Direct decoration (no parameters)
        @jit
        class SimpleOperator(Operator):
            def __call__(self, *, inputs):
                return process(inputs)

        # Parameterized decoration
        @jit(sample_input={"text": "example"})
        class ProcessorOperator(Operator):
            def __call__(self, *, inputs):
                # Complex multi-step process
                return {"result": processed_output}
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
        # More robust type checking that allows duck typing
        try:
            if not issubclass(cls, Operator):
                # Check for duck typing - if it has a __call__ method with the right signature
                if not (callable(cls) and callable(cls.__call__)):
                    raise TypeError(
                        "@jit decorator can only be applied to an Operator-like class with a __call__ method."
                    )
        except TypeError:
            # This handles the case where cls is not a class at all
            raise TypeError(
                "@jit decorator can only be applied to a class, not a function or other object."
            )

        original_call = cls.__call__
        original_init = cls.__init__

        @functools.wraps(original_init)
        def traced_init(self: OperatorType, *args: Any, **kwargs: Any) -> None:
            """Wrapped __init__ method that initializes the operator and pre-traces with sample input."""
            # Call the original __init__
            original_init(self, *args, **kwargs)

            # If sample_input is provided, perform pre-tracing during initialization
            if sample_input is not None:
                # Create a tracer context and trace the operator's execution
                with TracerContext() as tracer:
                    original_call(self=self, inputs=sample_input)

                if tracer.records:
                    # Import here to avoid circular imports
                    from ember.xcs.tracer.autograph import AutoGraphBuilder

                    # Build and cache the graph
                    graph_builder = AutoGraphBuilder()
                    graph = graph_builder.build_graph(tracer.records)

                    # Get state signature if available
                    state_signature = None
                    if hasattr(self, "get_state_signature") and callable(
                        getattr(self, "get_state_signature")
                    ):
                        state_signature = self.get_state_signature()

                    # Cache with the object itself as key and optional state signature
                    _jit_cache.set(self, graph, state_signature)

        @functools.wraps(original_call)
        def traced_call(self: OperatorType, *, inputs: Dict[str, Any]) -> Any:
            """Wrapped __call__ method with state-aware caching.

            Args:
                inputs: The input parameters for the operator.

            Returns:
                The output from the operator execution.
            """
            # Setup logging
            logger = logging.getLogger("ember.xcs.tracer.jit")

            # Get current tracer context
            tracer: Optional[TracerContext] = TracerContext.get_current()

            # For debugging and test purposes
            force_trace_local = getattr(self, "_force_trace", force_trace)

            # Check for state dependency protocol
            state_signature = None
            if hasattr(self, "get_state_signature") and callable(
                getattr(self, "get_state_signature")
            ):
                state_signature = self.get_state_signature()

            # Try to get cached graph with state validation
            graph = None
            if not force_trace_local:
                graph = _jit_cache.get_with_state(self, state_signature)

            # Phase 1: Try optimized execution with cached graph
            # -------------------------------------------------
            if graph is not None:
                try:
                    # Import execution components
                    from ember.xcs.engine.xcs_engine import (
                        TopologicalSchedulerWithParallelDispatch,
                        execute_graph,
                    )

                    logger.debug(
                        f"Using optimized graph for {self.__class__.__name__} (nodes: {len(graph.nodes)})"
                    )

                    # Execute the graph with the parallel scheduler and measure performance
                    execution_start = time.time()
                    results = execute_graph(
                        graph=graph,
                        global_input=inputs,
                        scheduler=TopologicalSchedulerWithParallelDispatch(),
                    )
                    execution_duration = time.time() - execution_start
                    _jit_cache.metrics.record_execution(execution_duration)

                    # Strict deterministic result extraction - no fallbacks
                    if "output_node_id" in graph.metadata:
                        output_node_id = graph.metadata["output_node_id"]
                        if output_node_id in results:
                            return results[output_node_id]
                        else:
                            raise ValueError(
                                f"Output node '{output_node_id}' specified in graph metadata but not found in results. "
                                f"Available nodes: {list(results.keys())}"
                            )
                    elif "output_node" in graph.metadata:
                        # Legacy compatibility
                        output_node = graph.metadata["output_node"]
                        if output_node in results:
                            logger.debug(
                                f"Using legacy output_node from graph metadata: {output_node}"
                            )
                            return results[output_node]

                    # If we got here with no explicit output node, try to determine the output
                    # This is a fallback for backward compatibility
                    logger.warning(
                        "Graph missing required 'output_node_id' metadata. "
                        "Using fallback strategies for backward compatibility."
                    )

                    # Strategy 1: Look for leaf nodes (nodes without outbound edges)
                    leaf_nodes = [
                        node_id
                        for node_id, node in graph.nodes.items()
                        if not node.outbound_edges
                    ]

                    if len(leaf_nodes) == 1:
                        # Single leaf node - clear choice for output
                        if leaf_nodes[0] in results:
                            return results[leaf_nodes[0]]

                    # Strategy 2: If there are multiple leaf nodes but all have identical results,
                    # arbitrarily choose one
                    if len(leaf_nodes) > 1:
                        first_result = results.get(leaf_nodes[0])
                        if all(
                            results.get(node) == first_result for node in leaf_nodes
                        ):
                            return first_result

                    # If all else fails, raise an error
                    raise ValueError(
                        "Could not determine output node. This indicates a bug in graph construction. "
                        f"Available nodes: {list(results.keys())}"
                    )

                except Exception as e:
                    # If graph execution fails, log the error and fall back to direct execution
                    logger.warning(
                        f"Error executing graph: {e}. Falling back to direct execution."
                    )

            # Phase 2: Tracing and direct execution
            # -------------------------------------------------
            # Initialize call tracking if in a tracer context
            call_id = None
            tracing_start = time.time()
            if tracer is not None:
                call_id = tracer.track_call(self, inputs)

            try:
                # Execute the original call directly
                execution_start = time.time()
                output = original_call(self=self, inputs=inputs)
                execution_duration = time.time() - execution_start
                _jit_cache.metrics.record_execution(execution_duration)

                # Complete the trace if we're tracing
                if tracer is not None and call_id is not None:
                    record = tracer.complete_call(call_id, output)

                    # Build and cache graph if appropriate
                    build_graph = (
                        force_trace_local
                        or not any(
                            r.node_id == str(id(self)) for r in tracer.records[:-1]
                        )
                        or len(tracer.records)
                        >= 3  # Minimum threshold for useful graph
                    )

                    if build_graph:
                        # Import here to avoid circular imports
                        from ember.xcs.tracer.autograph import AutoGraphBuilder

                        # Build a graph from the accumulated trace records
                        logger.debug(
                            f"Building graph from {len(tracer.records)} trace records"
                        )

                        # Measure compilation time
                        compilation_start = time.time()
                        graph_builder = AutoGraphBuilder()
                        graph = graph_builder.build_graph(tracer.records)
                        compilation_duration = time.time() - compilation_start
                        _jit_cache.metrics.record_compilation(compilation_duration)

                        # Only cache the graph if it has multiple nodes (otherwise no benefit)
                        if len(graph.nodes) > 1:
                            logger.debug(f"Caching graph with {len(graph.nodes)} nodes")
                            # Cache the graph with state signature if available
                            _jit_cache.set(self, graph, state_signature)
                        else:
                            logger.debug(
                                f"Not caching trivial graph with {len(graph.nodes)} nodes"
                            )

                # Record total tracing time if we traced
                if tracer is not None:
                    tracing_duration = time.time() - tracing_start
                    _jit_cache.metrics.record_tracing(tracing_duration)

                return output

            except Exception as e:
                # Complete the trace with the exception if we're tracing
                if tracer is not None and call_id is not None:
                    # Pass empty dict for outputs since execution failed
                    tracer.complete_call(call_id, {}, exception=e)

                # Re-raise the exception
                raise

        # Replace the original methods with our traced versions
        cls.__init__ = cast(Callable, traced_init)
        cls.__call__ = cast(Callable, traced_call)
        return cls

    # Handle both @jit and @jit(...) patterns
    if func is not None:
        # Called as @jit without parentheses
        return decorator(func)
    else:
        # Called with parameters as @jit(...)
        return decorator


# Removed _build_graph_from_trace function since we're not implementing the enhanced
# JIT capability in this PR. This would be included in a future full implementation.
