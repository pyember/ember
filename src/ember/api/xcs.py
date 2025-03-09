"""XCS API for Ember.

This module provides a clean interface for working with the XCS execution framework
in Ember, offering high-performance execution capabilities for computational graphs,
just-in-time tracing, and parallel execution transformations.

Examples:
    # Using JIT compilation for operators
    from ember.api import xcs
    
    @xcs.jit
    class MyOperator(Operator):
        def forward(self, *, inputs):
            # Complex computation here
            return result
    
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
    # Core execution framework
    xcs,                     # XCS core interface
    
    # Core functions
    jit,                     # Just-in-time compilation
    autograph,               # Automatic graph building
    execute,                 # Direct graph execution
    
    # Transforms
    vmap,                    # Vectorized mapping
    pmap,                    # Parallel mapping
    mesh_sharded,            # Sharded execution
    DeviceMesh,              # Device mesh for distribution
    PartitionSpec,           # Partition specification
    
    # Tracing
    TracerContext,           # Context for tracing
    TraceRecord,             # Record of traces
    TraceContextData,        # Trace context data
    
    # Types
    XCSExecutionOptions,     # Options for execution
    ExecutionResult,         # Result of execution
    JITOptions,              # Options for JIT
    TransformOptions,        # Options for transforms
)

__all__ = [
    # Core execution framework
    "xcs",
    
    # Core functions
    "jit",
    "autograph",
    "execute",
    
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