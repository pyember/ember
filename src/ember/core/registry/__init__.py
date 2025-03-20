"""Registry module for ember components."""

from __future__ import annotations

# Import EmberModel for easy access
from ember.core.types import EmberModel

# Import subpackages to make them available when importing registry
from . import model, operator, specification

__all__ = [
    "model",
    "operator",
    "specification",
    "EmberModel",
]
