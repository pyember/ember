"""
Core implementation of the XCS API.

This module provides the core implementation of the XCS API, offering a simplified
interface to the XCS functionality. It abstracts away the details of the underlying
implementation, providing a clean, intuitive interface for users.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast, Union

from ember.xcs.api.types import (
    XCSExecutionOptions,
    ExecutionResult,
    JITOptions,
    TransformOptions,
)
from ember.xcs.tracer.tracer_decorator import jit as raw_jit
from ember.xcs.tracer.autograph import AutoGraphBuilder
from ember.xcs.engine.xcs_engine import execute_graph
from ember.xcs.engine.execution_options import ExecutionOptions, execution_options
from ember.xcs.transforms.vmap import vmap as raw_vmap
from ember.xcs.transforms.pmap import pmap as raw_pmap, pjit as raw_pjit
from ember.xcs.transforms.mesh import (
    DeviceMesh,
    PartitionSpec,
    mesh_sharded as raw_mesh_sharded,
)
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.xcs_tracing import TraceRecord

# Type variable for operators
T = TypeVar("T")


class XCSAPI:
    """
    Main API class for XCS functionality.

    This class provides a unified interface to the XCS (eXecutable Computation System)
    functionality, abstracting away the details of the underlying implementation and
    providing a clean, intuitive interface for users.

    The design follows the Facade pattern, offering a simplified interface to the
    complex underlying system. It also implements the Adapter pattern to provide
    a consistent interface across different components.
    """

    def __init__(self) -> None:
        """Initialize the XCS API."""
        self._graph_builder = AutoGraphBuilder()

    def jit(
        self,
        target: Optional[Type[T]] = None,
        *,
        options: Optional[JITOptions] = None,
        **kwargs: Any,
    ) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
        """
        Just-In-Time compilation decorator for Ember Operators.

        The @jit decorator transforms Operator classes to automatically trace their execution
        and compile optimized execution plans. This brings significant performance benefits
        for complex operations and operator pipelines.

        Args:
            target: The operator class to decorate (when used directly)
            options: Configuration options for JIT compilation
            **kwargs: Additional options passed directly to the underlying implementation

        Returns:
            The decorated operator class or a decorator function

        Example:
            ```python
            from ember.xcs import xcs

            # As a decorator
            @xcs.jit
            class MyOperator(Operator):
                def forward(self, *, inputs):
                    # Complex logic here
                    return result

            # With options
            @xcs.jit(options=JITOptions(sample_input={"query": "Example"}))
            class MyOperator(Operator):
                ...
            ```
        """
        # Handle options
        opts = options or JITOptions()

        # Prepare kwargs for raw_jit
        jit_kwargs = {
            "sample_input": opts.sample_input,
            "force_trace": opts.force_trace,
            "recursive": opts.recursive,
            **kwargs,
        }

        # When used directly as @xcs.jit
        if target is not None:
            return raw_jit(**jit_kwargs)(target)

        # When used as @xcs.jit() or @xcs.jit(options=...)
        return lambda cls: raw_jit(**jit_kwargs)(cls)

    def autograph(self, records: List[TraceRecord]) -> XCSGraph:
        """
        Build an execution graph from trace records.

        This function analyzes trace records to identify dependencies between operators
        and constructs a graph that can be used for parallel execution.

        Args:
            records: List of trace records from execution

        Returns:
            An XCS graph for execution

        Example:
            ```python
            from ember.xcs import xcs
            from ember.xcs.tracer import TracerContext

            with TracerContext() as tracer:
                # Execute some operations
                op(inputs={"query": "Example"})

            # Build a graph from the trace records
            graph = xcs.autograph(tracer.records)
            ```
        """
        return self._graph_builder.build_graph(records=records)

    def execute(
        self,
        graph: XCSGraph,
        inputs: Dict[str, Any],
        options: Optional[XCSExecutionOptions] = None,
    ) -> ExecutionResult:
        """
        Execute an XCS graph with the given inputs.

        This function executes a graph with the given inputs, optionally using
        the provided execution options.

        Args:
            graph: The XCS graph to execute
            inputs: Input values for the graph
            options: Optional execution options

        Returns:
            The execution result, including outputs and timing information

        Example:
            ```python
            from ember.xcs import xcs

            # Create a graph
            graph = xcs.autograph(tracer.records)

            # Execute the graph
            result = xcs.execute(graph, inputs={"query": "Example"})
            print(result.outputs)
            ```
        """
        # Prepare execution options
        opts = options or XCSExecutionOptions()
        exec_options = execution_options(
            max_workers=opts.max_workers, timeout=opts.timeout
        )

        # Record start time
        import time

        start_time = time.time()

        # Execute graph
        outputs = execute_graph(graph=graph, inputs=inputs, options=exec_options)

        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Return the result
        return ExecutionResult(outputs=outputs, execution_time=execution_time)

    def vmap(
        self,
        fn: Callable[..., Any],
        options: Optional[TransformOptions] = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """
        Vectorize a function across its inputs.

        This function applies vectorization to a function, allowing it to efficiently
        process multiple inputs in parallel.

        Args:
            fn: The function to vectorize
            options: Configuration options for vectorization
            **kwargs: Additional options passed directly to the underlying implementation

        Returns:
            A vectorized version of the function

        Example:
            ```python
            from ember.xcs import xcs

            def process_item(item):
                return item * 2

            # Vectorize the function
            batch_process = xcs.vmap(process_item)

            # Process multiple items at once
            results = batch_process([1, 2, 3])  # [2, 4, 6]
            ```
        """
        # Handle options
        opts = options or TransformOptions()

        # Prepare kwargs for raw_vmap
        vmap_kwargs = {"in_axes": opts.in_axes, "out_axes": opts.out_axes, **kwargs}

        return raw_vmap(fn, **vmap_kwargs)

    def pmap(
        self,
        fn: Callable[..., Any],
        options: Optional[TransformOptions] = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """
        Parallelize a function across multiple devices.

        This function applies parallelization to a function, allowing it to efficiently
        process multiple inputs across different devices.

        Args:
            fn: The function to parallelize
            options: Configuration options for parallelization
            **kwargs: Additional options passed directly to the underlying implementation

        Returns:
            A parallelized version of the function

        Example:
            ```python
            from ember.xcs import xcs

            def process_item(item):
                return item * 2

            # Parallelize the function
            parallel_process = xcs.pmap(process_item)

            # Process multiple items across devices
            results = parallel_process([1, 2, 3])  # [2, 4, 6]
            ```
        """
        # Handle options
        opts = options or TransformOptions()

        # Prepare kwargs for raw_pmap
        pmap_kwargs = {
            "in_axes": opts.in_axes,
            "out_axes": opts.out_axes,
            "devices": opts.devices,
            **kwargs,
        }

        return raw_pmap(fn, **pmap_kwargs)

    def mesh_sharded(
        self,
        fn: Callable[..., Any],
        mesh: DeviceMesh,
        partition_spec: PartitionSpec,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """
        Apply mesh-based sharding to a function.

        This function applies mesh-based sharding to a function, allowing it to efficiently
        process inputs across a mesh of devices.

        Args:
            fn: The function to shard
            mesh: The device mesh to use
            partition_spec: The partition specification
            **kwargs: Additional options passed directly to the underlying implementation

        Returns:
            A sharded version of the function

        Example:
            ```python
            from ember.xcs import xcs

            # Create a device mesh
            mesh = xcs.DeviceMesh(devices=[0, 1, 2, 3], mesh_shape=(2, 2))

            # Create a partition spec
            pspec = xcs.PartitionSpec(0, 1)

            # Apply mesh-based sharding
            sharded_fn = xcs.mesh_sharded(fn, mesh, pspec)
            ```
        """
        return raw_mesh_sharded(fn, mesh, partition_spec, **kwargs)

    # Re-export classes for convenience
    DeviceMesh = DeviceMesh
    PartitionSpec = PartitionSpec
