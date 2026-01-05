"""User-facing API for Ember XCS.

Examples:
    >>> from ember.xcs import jit
    >>> @jit
    ... def square(x):
    ...     return x * x
"""

from ember.xcs.api import get_jit_stats, grad, jit, pmap, scan, vmap
from ember.xcs.compiler.analysis import (
    EffectRisk,
    OpDecision,
    OpKind,
    Traceability,
    analyze_operations_v2,
    explain,
)
from ember.xcs.config import Config, Presets
from ember.xcs.errors import XCSError
from ember.xcs.utils import register_ember_pytrees

register_ember_pytrees()

__all__ = [
    # Core transforms
    "jit",
    "vmap",
    "pmap",
    "scan",
    "grad",
    "get_jit_stats",
    # Configuration
    "Config",
    "Presets",
    # Analysis types
    "OpKind",
    "Traceability",
    "EffectRisk",
    "OpDecision",
    "analyze_operations_v2",
    "explain",
    # Errors
    "XCSError",
]
