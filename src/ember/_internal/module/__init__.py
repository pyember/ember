"""Module system primitives for Ember operators."""

from ember._internal.module.core import EmberModuleMeta, Module, field
from ember._internal.module.initialization import patch_module_class

__all__ = [
    "EmberModuleMeta",
    "Module",
    "field",
    "patch_module_class",
]
