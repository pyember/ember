"""
Core implementation of the XCS API.

Providing the implementation of the XCS API with a simplified
interface to XCS functionality. Abstracting away implementation details
for a clear interface.

This module serves as the entry point for using XCS capabilities.

Example:
    ```python
    from ember.xcs.api.core import XCSAPI

    # Creating an API instance
    xcs = XCSAPI()

    # Using the API to apply JIT optimization
    @xcs.jit(options={"sample_input": {"query": "test"}})
    class MyOperator(Operator):
        def forward(self, *, inputs):
            return {"result": process(inputs["query"])}

    # Executing an operator with tracing
    with xcs.tracing() as tracer:
        result = my_op(inputs={"query": "example"})

    # Building and executing a graph
    graph = xcs.autograph(tracer.records)
    final_result = xcs.execute(graph, inputs={"query": "test"})
    ```
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from ember.xcs.api.types import (
    ExecutionResult,
    JITOptions,
    TransformOptions,
    XCSExecutionOptions,
)
from ember.xcs.engine.execution_options import execution_options
from ember.xcs.engine.unified_engine import execute_graph
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.autograph import AutoGraphBuilder
from ember.xcs.jit import jit as raw_jit
from ember.xcs.tracer.xcs_tracing import TraceRecord
from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec
from ember.xcs.transforms.mesh import mesh_sharded as raw_mesh_sharded
from ember.xcs.transforms.pmap import pmap as raw_pmap
from ember.xcs.transforms.vmap import vmap as raw_vmap

# Type variable for operators
T = TypeVar("T")


class XCSAPI:
    """
    Main API class for XCS functionality.

    Providing a unified interface to the XCS (eXecutable Computation System)
    functionality, abstracting away the details of the implementation.

    This class provides a simplified interface to the system with a
    consistent interface across different components.

    Example:
        ```python
        from ember.xcs.api.core import XCSAPI

        # Creating API instance
        xcs = XCSAPI()

        # Using JIT compilation
        @xcs.jit
        class MyOperator(Operator):
            def forward(self, *, inputs):
                return {"result": process(inputs)}

        # Using vectorization
        batch_fn = xcs.vmap(single_item_fn)

        # Using parallelization
        parallel_fn = xcs.pmap(compute_intensive_fn)

        # Building and executing a graph
        graph = xcs.autograph(trace_records)
        result = xcs.execute(graph, inputs={"query": "test"})
        ```
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

        Transforming Operator classes to automatically trace their execution
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

            # Using as a direct decorator
            @xcs.jit
            class MyOperator(Operator):
                def forward(self, *, inputs):
                    # Complex logic here
                    return {"result": process(inputs["query"])}

            # Creating an instance and executing
            op = MyOperator()
            result = op(inputs={"query": "example"})

            # Using with advanced configuration options
            @xcs.jit(options=JITOptions(
                sample_input={"query": "Example"},
                force_trace=False,
                recursive=True
            ))
            class OptimizedOperator(Operator):
                def __init__(self):
                    self.sub_op1 = SubOperator1()
                    self.sub_op2 = SubOperator2()

                def forward(self, *, inputs):
                    # Multi-stage processing with optimized execution
                    intermediate = self.sub_op1(inputs=inputs)
                    return self.sub_op2(inputs=intermediate)
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
        Executing an XCS graph with the given inputs.

        Running a graph with the provided inputs, optionally using
        execution options to control parallelism and timeout settings.

        Args:
            graph: The XCS graph to execute
            inputs: Input values for the graph
            options: Optional execution options for controlling parallelism, workers, and timeout

        Returns:
            The execution result, including outputs and timing information

        Example:
            ```python
            from ember.xcs import xcs
            from ember.xcs.api.types import XCSExecutionOptions

            # Creating a graph from trace records
            graph = xcs.autograph(tracer.records)

            # Setting execution options
            exec_options = XCSExecutionOptions(
                max_workers=4,     # Using up to 4 parallel workers
                timeout=10000      # Setting a 10-second timeout
            )

            # Executing the graph with options
            result = xcs.execute(
                graph,
                inputs={"query": "Example", "temperature": 0.7},
                options=exec_options
            )

            # Accessing results and metrics
            print(f"Outputs: {result.outputs}")
            print(f"Execution time: {result.execution_time:.2f}s")
            ```
        """
        # Prepare execution options
        opts = options or XCSExecutionOptions()
        
        # Convert API options to engine options
        from ember.xcs.engine.unified_engine import ExecutionOptions
        engine_options = ExecutionOptions(
            scheduler_type="parallel",  # Always use parallel execution in the API for performance
            max_workers=opts.max_workers,
            timeout_seconds=opts.timeout / 1000 if opts.timeout else None,
            collect_metrics=True
        )

        # Record start time
        import time
        start_time = time.time()

        # Execute graph with the unified engine
        outputs = execute_graph(
            graph=graph, 
            global_input=inputs,
            options=engine_options
        )

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
