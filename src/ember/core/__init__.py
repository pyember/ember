"""Core components of the Ember framework."""

from __future__ import annotations

# Absolute imports
try:
    # Try the normal import path first
    from ember.core import registry
    from ember.core.app_context import EmberContext
    from ember.core.exceptions import ConfigurationError, EmberError, ValidationError
    from ember.core.non import Sequential
except ImportError:
    # Fall back to src.ember if the regular imports fail
    from ember.core import registry
    from ember.core.app_context import EmberContext
    from ember.core.exceptions import ConfigurationError, EmberError, ValidationError
    from ember.core.non import Sequential

__all__ = [
    "EmberContext",
    "EmberError",
    "ValidationError",
    "ConfigurationError",
    "registry",
    "Sequential",
]
