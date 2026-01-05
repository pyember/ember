"""Context management utilities for Ember's runtime.

This package exposes the `EmberContext` entry point and helper functions for
retrieving or scoping the active configuration context.
"""

from ember._internal.context.metrics import MetricsContext
from ember._internal.context.runtime import EmberContext, current_context

__all__ = ["EmberContext", "MetricsContext", "current_context"]
