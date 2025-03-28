"""
Simplified XCS imports for tests.

This module provides access to the XCS functions and classes through the simplified import structure
specifically for testing the simplified import mechanism.
"""

from ember.xcs.transforms.mesh import DeviceMesh, mesh_sharded
from ember.xcs.transforms.pmap import pmap

# Import transform functions directly from their implementation modules
from ember.xcs.transforms.vmap import vmap


# Minimal stubs for testing the simplified imports
class XCSExecutionOptions:
    """Implementation of XCSExecutionOptions for testing."""

    def __init__(self, max_workers=4, timeout=None, **kwargs):
        self.max_workers = max_workers
        self.timeout = timeout
        self.__dict__.update(kwargs)


class ExecutionResult:
    """Implementation of ExecutionResult for testing."""

    def __init__(self, outputs=None, execution_time=0.0, **kwargs):
        self.outputs = outputs or {}
        self.execution_time = execution_time
        self.__dict__.update(kwargs)


class JITOptions:
    """Implementation of JITOptions for testing."""

    def __init__(self, sample_input=None, force_trace=False, recursive=False, **kwargs):
        self.sample_input = sample_input or {"query": "test"}
        self.force_trace = force_trace
        self.recursive = recursive
        self.__dict__.update(kwargs)


class TransformOptions:
    """Implementation of TransformOptions for testing."""

    def __init__(self, in_axes=0, out_axes=0, devices=None, **kwargs):
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.devices = devices
        self.__dict__.update(kwargs)


class TracerContext:
    """Implementation of TracerContext for testing."""

    def __init__(self, records=None, **kwargs):
        self.records = records or []
        self.__dict__.update(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TraceRecord:
    """Implementation of TraceRecord for testing."""

    def __init__(
        self, operator_name="", node_id="", inputs=None, outputs=None, **kwargs
    ):
        self.operator_name = operator_name
        self.node_id = node_id
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.__dict__.update(kwargs)


class XCSAPI:
    """Implementation of XCSAPI for testing."""

    def __init__(self):
        pass

    def jit(self, func=None, **kwargs):
        """JIT decorator implementation."""
        return func if func is not None else lambda f: f

    def autograph(self, func=None, **kwargs):
        """Autograph decorator implementation."""
        return func if func is not None else lambda f: f

    def execute(self, graph=None, **kwargs):
        """Execute implementation."""
        return ExecutionResult()

    def vmap(self, func=None, **kwargs):
        """Vectorized map implementation."""
        return vmap(func) if func is not None else lambda f: vmap(f)

    def pmap(self, func=None, **kwargs):
        """Parallel map implementation."""
        return pmap(func) if func is not None else lambda f: pmap(f)

    def mesh_sharded(self, func=None, **kwargs):
        """Mesh-sharded implementation."""
        mesh = kwargs.get("mesh", DeviceMesh())
        return (
            mesh_sharded(func, mesh)
            if func is not None
            else lambda f: mesh_sharded(f, mesh)
        )


# Create instance of the API
xcs = XCSAPI()


# Define the function interfaces matching the simplified imports
def jit(func=None, **kwargs):
    """JIT decorator matching the simplified import."""
    return xcs.jit(func, **kwargs)


def autograph(records=None, **kwargs):
    """Autograph function matching the simplified import."""
    return xcs.autograph(records, **kwargs)


def execute(graph=None, inputs=None, **kwargs):
    """Execute function matching the simplified import."""
    inputs = inputs or {}
    return xcs.execute(graph, inputs, **kwargs)
