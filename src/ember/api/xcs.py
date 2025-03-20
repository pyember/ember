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

    # Using structural JIT for optimized execution
    @xcs.structural_jit(execution_strategy="parallel")
    class CompositeOperator(Operator):
        def __init__(self):
            self.op1 = SubOperator1()
            self.op2 = SubOperator2()

        def __call__(self, *, inputs):
            # Multi-step computation automatically parallelized
            result1 = self.op1(inputs=inputs)
            result2 = self.op2(inputs=inputs)
            return combine(result1, result2)

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
from ember.xcs import DeviceMesh  # Device mesh for distribution
from ember.xcs import ExecutionResult  # Result of execution
from ember.xcs import JITOptions  # Options for JIT
from ember.xcs import PartitionSpec  # Partition specification
from ember.xcs import TraceContextData  # Trace context data
from ember.xcs import TracerContext  # Context for tracing
from ember.xcs import TraceRecord  # Record of traces
from ember.xcs import TransformOptions  # Options for transforms
from ember.xcs import XCSExecutionOptions  # Options for execution
from ember.xcs import autograph  # Automatic graph building
from ember.xcs import execute  # Direct graph execution
from ember.xcs import jit  # Just-in-time compilation
from ember.xcs import mesh_sharded  # Sharded execution
from ember.xcs import pmap  # Parallel mapping
from ember.xcs import structural_jit  # Structural JIT compilation
from ember.xcs import vmap  # Vectorized mapping
from ember.xcs import (
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
