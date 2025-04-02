"""Execution engine for XCS computation graphs.

Provides the core execution engine for running computational graphs with
different scheduling strategies and optimizations. This module forms the
foundation for the XCS system's execution capabilities.

The engine follows a unified architecture with clean separation of concerns
between graph construction, scheduling, and execution.
"""

# Core execution options API
from ember.xcs.engine.execution_options import (
    ExecutionOptions,
    execution_options,
)

# Core engine functionality
from ember.xcs.engine.unified_engine import (
    execute_graph,
    GraphExecutor,
    ExecutionMetrics,
)

# All scheduler functionality is in the schedulers package
from ember.xcs.schedulers import create_scheduler

__all__ = [
    # Execution options
    "ExecutionOptions",
    "execution_options",
    
    # Engine core
    "execute_graph",
    "GraphExecutor",
    "ExecutionMetrics",
    "create_scheduler",
]