"""
Common utilities and shared structures for the XCS module.

This package contains core data structures and utilities used across
different components of the XCS system, including execution plans, 
shared type definitions, and helper functions.
"""

from ember.xcs.common.plans import XCSPlan, XCSTask, ExecutionResult

__all__ = [
    "XCSPlan",
    "XCSTask",
    "ExecutionResult",
]