"""
Re-export for key modules from the operator package to simplify downstream imports.
"""

from __future__ import annotations

# Absolute imports for core operator components
from ember.core.registry.operator.base import Operator

__all__ = [
    "Operator",
]
