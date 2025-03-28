"""XCS API for Ember.

This module provides a clean interface for working with the XCS execution framework
in Ember, offering high-performance execution capabilities for computational graphs,
just-in-time tracing, and parallel execution transformations.

Examples:
    # Basic JIT compilation (defaults to trace-based)
    from ember.api import xcs

    @xcs.jit
    class MyOperator(Operator):
        def forward(self, *, inputs):
            # Complex computation here
            return result

    # Using trace-based JIT with specific options
    @xcs.jit.trace(sample_input={"query": "test"})
    class TracedOperator(Operator):
        def forward(self, *, inputs):
            # Complex computation here
            return result

    # Using structural JIT for optimized parallel execution
    @xcs.jit.structural(execution_strategy="parallel")
    class ParallelOperator(Operator):
        def __init__(self):
            self.op1 = SubOperator1()
            self.op2 = SubOperator2()

        def forward(self, *, inputs):
            # Multi-step computation automatically parallelized
            result1 = self.op1(inputs=inputs)
            result2 = self.op2(inputs=inputs)
            return combine(result1, result2)

    # Legacy syntax (still supported but not recommended)
    @xcs.structural_jit(execution_strategy="parallel")
    class LegacyOperator(Operator):
        # implementation...
        pass

    # Using vectorized mapping
    @xcs.vmap(in_axes=(0, None))
    def process_batch(inputs, model):
        return model(inputs)

    # Using parallel execution
    @xcs.pmap
    def parallel_process(inputs):
        return heavy_computation(inputs)
"""

# Import from the implementation
from ember.xcs import (
    DeviceMesh,  # Device mesh for distribution
    ExecutionResult,  # Result of execution
    JITOptions,  # Options for JIT
    PartitionSpec,  # Partition specification
    TraceContextData,  # Trace context data
    TracerContext,  # Context for tracing
    TraceRecord,  # Record of traces
    TransformOptions,  # Options for transforms
    XCSExecutionOptions,  # Options for execution
    autograph,  # Automatic graph building
    execute,  # Direct graph execution
    jit,  # Just-in-time compilation
    mesh_sharded,  # Sharded execution
    pmap,  # Parallel mapping
    structural_jit,  # Structural JIT compilation
    vmap,  # Vectorized mapping
    xcs,  # Core execution framework; Core functions; Transforms; Tracing; Types; XCS core interface
)

# Import execution options directly - essential for controlling parallel execution
from ember.xcs.engine.execution_options import execution_options

__all__ = [
    # Core execution framework
    "xcs",
    # Core functions
    "jit",
    "structural_jit",  # Advanced structure-aware JIT
    "autograph",
    "execute",
    "execution_options",  # Essential for execution control
    # Transforms
    "vmap",
    "pmap",
    "mesh_sharded",
    "DeviceMesh",
    "PartitionSpec",
    # Tracing
    "TracerContext",
    "TraceRecord",
    "TraceContextData",
    # Types
    "XCSExecutionOptions",
    "ExecutionResult",
    "JITOptions",
    "TransformOptions",
]
