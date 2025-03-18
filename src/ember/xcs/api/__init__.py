"""
XCS API

This module provides a unified interface to the XCS (Accelerated Compound Systems) 
functionality. It follows the ember pattern of providing a simplified, intuitive 
interface on top of the core implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union

from ember.xcs.api.core import XCSAPI
from ember.xcs.api.types import (
    XCSExecutionOptions,
    ExecutionResult,
    GraphBuilder,
    JITOptions,
    TransformOptions,
)

# Create a singleton instance
xcs = XCSAPI()

# Re-export types for convenience
__all__ = [
    "xcs",
    "XCSExecutionOptions",
    "ExecutionResult",
    "GraphBuilder",
    "JITOptions",
    "TransformOptions",
]
