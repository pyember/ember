"""Registry module for ember components."""

from __future__ import annotations

# Import subpackages to make them available when importing registry
from . import model
from . import operator
from . import prompt_specification

# Import EmberModel for easy access
from ember.core.types import EmberModel

__all__ = [
    "model",
    "operator",
    "prompt_specification",
    "EmberModel",
]
