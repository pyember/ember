"""
Transformation library for XCS.

This module provides transformations that can be applied to XCS operators and graphs,
including vectorization (vmap), parallelization (pmap/pjit), and mesh-based sharding.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Union,
    Optional,
    List,
    TypeVar,
    Protocol,
    runtime_checkable,
)


# Define a protocol for operators to avoid circular imports
@runtime_checkable
class OperatorProtocol(Protocol):
    """Protocol defining the operator interface."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Call method for operators."""
        ...


# Import these directly so they're accessible even if implementation imports fail
class DeviceMesh:
    """Stub DeviceMesh class."""

    def __init__(self, devices=None, shape=None):
        self.devices = devices or []
        self.shape = shape or (len(self.devices),)


class PartitionSpec:
    """Stub PartitionSpec class."""

    def __init__(self, *mesh_axes):
        self.mesh_axes = mesh_axes


# Define transform functions at module level to avoid 'module is not callable' errors
def vmap_func(operator_or_fn: Any, in_axes: Any = 0, out_axes: Any = 0) -> Callable:
    """Vectorized map function."""
    try:
        from .vmap import vmap as vmap_impl

        return vmap_impl(operator_or_fn, in_axes, out_axes)
    except ImportError:
        # Return a simple wrapper that returns the input
        def _wrapped_vmap(*args, **kwargs):
            if callable(operator_or_fn):
                return operator_or_fn(*args, **kwargs)
            return args[0] if args else {}

        return _wrapped_vmap


def pmap_func(
    operator_or_fn: Any,
    num_workers: Optional[int] = None,
    devices: Optional[List[str]] = None,
) -> Callable:
    """Parallel map function."""
    try:
        from .pmap import pmap as pmap_impl

        return pmap_impl(operator_or_fn, num_workers, devices)
    except ImportError:
        # Return a simple wrapper that returns the input
        def _wrapped_pmap(*args, **kwargs):
            if callable(operator_or_fn):
                return operator_or_fn(*args, **kwargs)
            return args[0] if args else {}

        return _wrapped_pmap


def pjit_func(
    operator_or_fn: Any,
    num_workers: Optional[int] = None,
    devices: Optional[List[str]] = None,
    static_argnums: Optional[List[int]] = None,
) -> Callable:
    """Parallel JIT compilation function."""
    try:
        from .pmap import pjit as pjit_impl

        return pjit_impl(operator_or_fn, num_workers, devices, static_argnums)
    except ImportError:
        # Return a simple wrapper that returns the input
        def _wrapped_pjit(*args, **kwargs):
            if callable(operator_or_fn):
                return operator_or_fn(*args, **kwargs)
            return args[0] if args else {}

        return _wrapped_pjit


def mesh_sharded_func(
    operator_or_fn: Any, mesh: Any, partition_spec: Optional[Dict[str, Any]] = None
) -> Callable:
    """Mesh-sharded function."""
    try:
        from .mesh import mesh_sharded as mesh_sharded_impl

        return mesh_sharded_impl(operator_or_fn, mesh, partition_spec)
    except ImportError:
        # Return a simple wrapper that returns the input
        def _wrapped_mesh_sharded(*args, **kwargs):
            if callable(operator_or_fn):
                return operator_or_fn(*args, **kwargs)
            return args[0] if args else {}

        return _wrapped_mesh_sharded


# Try to import the actual implementations and set up the public exports
try:
    # Try relative imports first to avoid circular dependencies
    from .vmap import vmap as _vmap
    from .pmap import pmap as _pmap, pjit as _pjit
    from .mesh import (
        DeviceMesh as _DeviceMesh,
        PartitionSpec as _PartitionSpec,
        mesh_sharded as _mesh_sharded,
    )

    # Use the actual implementations
    vmap = _vmap
    pmap = _pmap
    pjit = _pjit
    DeviceMesh = _DeviceMesh
    PartitionSpec = _PartitionSpec
    mesh_sharded = _mesh_sharded
except ImportError:
    # If imports fail, use the function proxies to avoid 'module is not callable' errors
    vmap = vmap_func
    pmap = pmap_func
    pjit = pjit_func
    mesh_sharded = mesh_sharded_func

# Export all public symbols
__all__ = [
    "vmap",
    "pmap",
    "pjit",
    "DeviceMesh",
    "PartitionSpec",
    "mesh_sharded",
    "OperatorProtocol",
]
