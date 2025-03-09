"""Registry module for ember components."""

from __future__ import annotations

# Import subpackages to make them available when importing registry
from . import model
from . import operator
from . import specification

# Import EmberModel for easy access
from ember.core.types import EmberModel

__all__ = [
    "model",
    "operator",
    "specification",
    "EmberModel",
]
