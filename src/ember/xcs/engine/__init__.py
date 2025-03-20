"""Ember XCS Engine Package.

This package provides the execution engine and scheduler interfaces for Ember XCS.
"""

from ember.xcs.engine.execution_options import ExecutionOptions, execution_options
from ember.xcs.engine.xcs_engine import (
    IScheduler,
    TopologicalScheduler,
    TopologicalSchedulerWithParallelDispatch,
    execute_graph,
)

__all__ = [
    "execute_graph",
    "execution_options",
    "ExecutionOptions",
    "IScheduler",
    "TopologicalScheduler",
    "TopologicalSchedulerWithParallelDispatch",
]
