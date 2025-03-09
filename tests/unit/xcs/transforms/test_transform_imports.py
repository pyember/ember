"""
Direct imports for transform functions.

This module imports transform functions directly from their respective modules 
to ensure they're accessible without going through the transforms module.
"""

# Import directly from implementation modules
from ember.xcs.transforms.vmap import vmap
from ember.xcs.transforms.pmap import pmap, pjit
from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec, mesh_sharded
from ember.xcs.transforms import OperatorProtocol

__all__ = [
    "vmap",
    "pmap",
    "pjit",
    "DeviceMesh", 
    "PartitionSpec",
    "mesh_sharded",
    "OperatorProtocol"
]