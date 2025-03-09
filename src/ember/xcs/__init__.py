"""
XCS: eXecutable Computation System

The XCS module provides a high-performance distributed execution framework for 
computational graphs. It implements a directed acyclic graph (DAG) architecture 
for operator composition, intelligent scheduling, and just-in-time tracing.

Key components:
- Graph: DAG-based intermediate representation (IR) for defining operator pipelines
- Engine: Concurrent execution scheduler with automatic dependency resolution
- Tracer: JIT tracing system that creates execution graphs from function calls
- Transforms: Higher-order operations for batching, parallelization, and more

This module implements the Facade pattern to provide a clean, simplified interface
to the underlying XCS functionality while abstracting away implementation details.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast, overload
from typing import Protocol, runtime_checkable
from contextlib import contextmanager
import functools
import sys
import warnings
from types import ModuleType

# Type variables for generic function signatures
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# Define protocol types for XCS components
@runtime_checkable
class XCSGraph(Protocol):
    """Protocol defining the interface for XCS computation graphs."""
    
    def add_node(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """Add a node to the computation graph.
        
        Args:
            name: The name of the node
            func: The function to execute at this node
            *args: Positional arguments to the function
            **kwargs: Keyword arguments to the function
            
        Returns:
            The node ID as a string
        """
        ...
    
    def execute(self, output_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute the computation graph.
        
        Args:
            output_nodes: Optional list of node IDs to return results for
            
        Returns:
            A dictionary mapping node IDs to their execution results
        """
        ...

@runtime_checkable
class ExecutionOptions(Protocol):
    """Protocol defining the interface for execution options."""
    
    @property
    def parallel(self) -> bool:
        """Whether to execute the graph in parallel."""
        ...
    
    @parallel.setter
    def parallel(self, value: bool) -> None:
        """Set whether to execute the graph in parallel."""
        ...
    
    @property
    def max_workers(self) -> Optional[int]:
        """Maximum number of worker threads/processes."""
        ...
    
    @max_workers.setter
    def max_workers(self, value: Optional[int]) -> None:
        """Set the maximum number of worker threads/processes."""
        ...

# Try to import actual implementations, with fallbacks for testing environments
try:
    from .tracer.autograph import autograph as _autograph
    from .tracer.tracer_decorator import jit as _jit
    from .transforms.vmap import vmap as _vmap
    from .transforms.pmap import pmap as _pmap
    from .transforms.mesh import mesh_sharded as _mesh_sharded
    from .engine.xcs_engine import execute as _execute
    from .engine.execution_options import ExecutionOptions as _ExecutionOptions
    from .graph.xcs_graph import XCSGraph as _XCSGraph
    
    # Additional imports needed by api.xcs
    from .transforms.mesh import DeviceMesh, PartitionSpec
    from .tracer._context_types import TracerContext, TraceRecord, TraceContextData
    from .engine.execution_options import XCSExecutionOptions, JITOptions, TransformOptions, ExecutionResult
    
    _IMPORTS_AVAILABLE = True
    
except ImportError as e:
    _IMPORTS_AVAILABLE = False
    warnings.warn(f"XCS functionality partially unavailable: {e}. Using stub implementations.")
    
    # Stub implementations for testing environments
    @contextmanager
    def _autograph(*args: Any, **kwargs: Any) -> Any:
        """Stub implementation of autograph for testing."""
        yield _StubGraph()
    
    def _jit(func: Callable[..., T]) -> Callable[..., T]:
        """Stub implementation of jit for testing."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)
        return wrapper
    
    def _vmap(func: Callable[..., T]) -> Callable[..., List[T]]:
        """Stub implementation of vmap for testing."""
        @functools.wraps(func)
        def wrapper(xs: List[Any], *args: Any, **kwargs: Any) -> List[T]:
            return [func(x, *args, **kwargs) for x in xs]
        return wrapper
    
    def _pmap(func: Callable[..., T]) -> Callable[..., List[T]]:
        """Stub implementation of pmap for testing."""
        @functools.wraps(func)
        def wrapper(xs: List[Any], *args: Any, **kwargs: Any) -> List[T]:
            return [func(x, *args, **kwargs) for x in xs]
        return wrapper
    
    def _mesh_sharded(func: Callable[..., T]) -> Callable[..., T]:
        """Stub implementation of mesh_sharded for testing."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)
        return wrapper
    
    def _execute(graph: Any, output_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Stub implementation of execute for testing."""
        return {}
    
    class _ExecutionOptions:
        """Stub implementation of ExecutionOptions for testing."""
        
        def __init__(self) -> None:
            self._parallel = False
            self._max_workers = None
            
        @property
        def parallel(self) -> bool:
            return self._parallel
            
        @parallel.setter
        def parallel(self, value: bool) -> None:
            self._parallel = value
            
        @property
        def max_workers(self) -> Optional[int]:
            return self._max_workers
            
        @max_workers.setter
        def max_workers(self, value: Optional[int]) -> None:
            self._max_workers = value
    
    class _StubGraph:
        """Stub implementation of XCSGraph for testing."""
        
        def add_node(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
            """Add a node to the stub computation graph."""
            return name
            
        def execute(self, output_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
            """Execute the stub computation graph."""
            return {}
            
    # Stub implementations for additional imports
    DeviceMesh = object
    PartitionSpec = object
    class TracerContext: pass
    class TraceRecord: pass
    class TraceContextData: pass
    class XCSExecutionOptions: pass
    class JITOptions: pass
    class TransformOptions: pass
    class ExecutionResult: pass

# Public API exports with proper type annotations
@contextmanager
def autograph(*args: Any, **kwargs: Any) -> Any:
    """Context manager for automatic graph building.
    
    Creates a computation graph context where operations are automatically
    recorded as graph nodes rather than being executed immediately.
    
    Args:
        *args: Positional arguments to pass to the underlying implementation
        **kwargs: Keyword arguments to pass to the underlying implementation
        
    Yields:
        An XCSGraph object that can be used to execute the recorded operations
        
    Example:
        ```python
        with autograph() as g:
            x = foo(10)
            y = bar(x)
            
        results = g.execute()
        ```
    """
    with _autograph(*args, **kwargs) as graph:
        yield graph

def jit(func: Callable[..., T]) -> Callable[..., T]:
    """Just-In-Time compilation decorator for functions.
    
    Applies JIT compilation to the decorated function, which can
    significantly improve performance for repeated executions.
    
    Args:
        func: The function to be JIT-compiled
        
    Returns:
        A wrapped function that will be JIT-compiled on first execution
        
    Example:
        ```python
        @jit
        def my_function(x: int) -> int:
            return x * 2
        ```
    """
    return _jit(func)

def vmap(func: Callable[..., T]) -> Callable[..., List[T]]:
    """Vectorized mapping transformation.
    
    Transforms a function that operates on single elements into one that
    operates on vectors (lists) of elements in a vectorized manner.
    
    Args:
        func: The function to transform
        
    Returns:
        A function that applies the original function to each element of input vectors
        
    Example:
        ```python
        def add_one(x: int) -> int:
            return x + 1
            
        batch_add_one = vmap(add_one)
        results = batch_add_one([1, 2, 3])  # [2, 3, 4]
        ```
    """
    return _vmap(func)

def pmap(func: Callable[..., T]) -> Callable[..., List[T]]:
    """Parallel mapping transformation.
    
    Transforms a function that operates on single elements into one that
    operates on vectors (lists) of elements in parallel.
    
    Args:
        func: The function to transform
        
    Returns:
        A function that applies the original function to each element of input vectors in parallel
        
    Example:
        ```python
        def process_item(x: dict) -> dict:
            # Some expensive processing
            return processed_x
            
        parallel_processor = pmap(process_item)
        results = parallel_processor(items)
        ```
    """
    return _pmap(func)

def mesh_sharded(func: Callable[..., T]) -> Callable[..., T]:
    """Mesh sharding transformation.
    
    Transforms a function to execute in a distributed mesh environment,
    sharding computations across available resources.
    
    Args:
        func: The function to transform
        
    Returns:
        A function that executes the original function in a distributed manner
        
    Example:
        ```python
        @mesh_sharded
        def large_matrix_operation(matrix: np.ndarray) -> np.ndarray:
            # Operation on a large matrix
            return result
        ```
    """
    return _mesh_sharded(func)

def execute(graph: XCSGraph, output_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute a computation graph.
    
    Args:
        graph: The computation graph to execute
        output_nodes: Optional list of node IDs to return results for
        
    Returns:
        A dictionary mapping node IDs to their execution results
        
    Example:
        ```python
        with autograph() as g:
            x = foo(10)
            y = bar(x)
            
        results = execute(g, output_nodes=['y'])
        ```
    """
    return _execute(graph=graph, output_nodes=output_nodes)

# Self-reference for import in api.xcs.py
xcs = sys.modules[__name__]

# Export types for user convenience
XCSGraph = _XCSGraph if _IMPORTS_AVAILABLE else _StubGraph
# For backward compatibility with the original code
if not _IMPORTS_AVAILABLE:
    XCSExecutionOptions = _ExecutionOptions

# Exported symbols for __all__
__all__ = [
    'xcs',
    'autograph',
    'jit',
    'vmap',
    'pmap',
    'mesh_sharded',
    'execute',
    'XCSGraph',
    'ExecutionOptions',
    'DeviceMesh',
    'PartitionSpec',
    'TracerContext',
    'TraceRecord',
    'TraceContextData',
    'XCSExecutionOptions',
    'JITOptions',
    'TransformOptions',
    'ExecutionResult',
]

# Export version information if available
try:
    from . import __version__
    __all__.append('__version__')
except ImportError:
    pass