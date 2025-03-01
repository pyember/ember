"""Ember XCS Engine Package.

This package provides the execution engine and scheduler interfaces for Ember XCS.
"""

from ember.xcs.engine.execution_options import execution_options, ExecutionOptions
from ember.xcs.engine.xcs_engine import (
    execute_graph,
    IScheduler,
    TopologicalSchedulerWithParallelDispatch,
)

__all__ = [
    "execute_graph",
    "execution_options",
    "ExecutionOptions",
    "IScheduler",
    "TopologicalSchedulerWithParallelDispatch",
]
