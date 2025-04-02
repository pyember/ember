"""XCS (Accelerated Compound Systems) for Ember.

Provides a computational graph-based system for building, optimizing, and
executing complex operator pipelines with automatic parallelization and
optimization.
"""

# === Core JIT System ===
from ember.xcs.jit import jit, JITMode, JITCache, get_jit_stats, explain_jit_selection

# === Tracing Infrastructure ===
from ember.xcs.tracer.xcs_tracing import TracerContext, TraceRecord
from ember.xcs.tracer._context_types import TraceContextData
from ember.xcs.tracer.autograph import AutoGraphBuilder, autograph

# === API Types ===
from ember.xcs.api.types import (
    JITOptions,
    XCSExecutionOptions,
    ExecutionResult as APIExecutionResult,
    TransformOptions,
)

# === Graph Representation ===
from ember.xcs.graph.xcs_graph import XCSGraph, XCSNode
from ember.xcs.graph.dependency_analyzer import DependencyAnalyzer
from ember.xcs.graph.graph_builder import GraphBuilder, EnhancedTraceGraphBuilder

# === Execution Engine ===
from ember.xcs.engine.unified_engine import (
    execute_graph,
    GraphExecutor,
    ExecutionMetrics,
)
from ember.xcs.engine.execution_options import (
    ExecutionOptions,
    execution_options,
)
from ember.xcs.common.plans import XCSPlan, XCSTask, ExecutionResult

# === Transformations ===
from ember.xcs.transforms.transform_base import (
    BaseTransformation,
    BatchingOptions,
    ParallelOptions,
    TransformError,
    compose,
)
from ember.xcs.transforms.vmap import vmap
from ember.xcs.transforms.pmap import pmap, pjit
from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec, mesh_sharded

# === Scheduler System ===
from ember.xcs.schedulers.base_scheduler import BaseScheduler
from ember.xcs.schedulers.factory import create_scheduler
from ember.xcs.schedulers.unified_scheduler import (
    NoOpScheduler,
    ParallelScheduler,
    SequentialScheduler,
    TopologicalScheduler,
    WaveScheduler,
)

__all__ = [
    # Core JIT system
    "jit",
    "JITMode",
    "get_jit_stats",
    "JITCache",
    "explain_jit_selection",
    
    # API Types
    "JITOptions",
    "XCSExecutionOptions",
    "APIExecutionResult",
    "TransformOptions",
    
    # Tracing infrastructure
    "TracerContext",
    "TraceRecord",
    "TraceContextData",
    "AutoGraphBuilder",
    "autograph",
    
    # Graph representation
    "XCSGraph",
    "XCSNode",
    "DependencyAnalyzer",
    "GraphBuilder",
    "EnhancedTraceGraphBuilder",
    
    # Execution engine
    "execute_graph",
    "ExecutionOptions",
    "execution_options",
    "GraphExecutor", 
    "ExecutionMetrics",
    "XCSPlan",
    "XCSTask",
    "ExecutionResult",
    
    # Scheduler system
    "BaseScheduler",
    "NoOpScheduler",
    "ParallelScheduler",
    "SequentialScheduler",
    "TopologicalScheduler",
    "WaveScheduler",
    "create_scheduler",
    
    # Transformations
    "vmap",
    "pmap",
    "pjit",
    "DeviceMesh",
    "PartitionSpec",
    "mesh_sharded",
    "compose",
    "TransformError",
    "BaseTransformation",
    "BatchingOptions",
    "ParallelOptions",
]