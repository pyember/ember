"""
Transformation library for XCS.

This module provides transformations that can be applied to XCS operators and graphs,
including vectorization (vmap), parallelization (pmap/pjit), and mesh-based sharding.
"""

from ember.xcs.transforms.vmap import vmap
from ember.xcs.transforms.pmap import pmap, pjit
from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec, mesh_sharded

__all__ = [
    "vmap",
    "pmap",
    "pjit",
    "DeviceMesh",
    "PartitionSpec",
    "mesh_sharded",
]