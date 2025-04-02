"""Unified JIT compilation system for XCS.

Provides Just-In-Time compilation strategies for optimizing operator execution
through tracing, structural analysis, and enhanced dependency tracking.
"""

from typing import Any, Callable, Dict, Optional

# Import JIT modes
from ember.xcs.jit.modes import JITMode

# Core JIT decorator - these need to be imported after JITMode to avoid circular imports
from ember.xcs.jit.core import jit, get_jit_stats, explain_jit_selection

# JIT caching system
from ember.xcs.jit.cache import JITCache, get_cache

__all__ = [
    # Core JIT functionality
    "jit",
    "JITMode",
    "JITCache",
    "get_jit_stats",
    "explain_jit_selection",
    "get_cache",
]