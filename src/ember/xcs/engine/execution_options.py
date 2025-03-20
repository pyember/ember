"""
Execution options for XCS graph execution.

This module provides configuration options for graph execution, allowing
customization of dispatching strategy, parallelism, and device selection.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Union


@dataclasses.dataclass
class ExecutionOptions:
    """Options for XCS graph execution.

    Attributes:
        use_parallel: Whether to use parallel execution where possible.
        max_workers: Maximum number of worker threads for parallel execution.
        device_strategy: Strategy for device selection ('auto', 'cpu', 'gpu', etc.).
        enable_caching: Whether to cache intermediate results.
        trace_execution: Whether to trace execution for debugging.
        timeout_seconds: Maximum execution time in seconds before timeout.
    """

    use_parallel: bool = True
    max_workers: Optional[int] = None
    device_strategy: str = "auto"
    enable_caching: bool = False
    trace_execution: bool = False
    timeout_seconds: Optional[float] = None


# Global default execution options
execution_options = ExecutionOptions()


def set_execution_options(**kwargs: Any) -> None:
    """Update the global execution options.

    Args:
        **kwargs: Keyword arguments to update execution options.
            Valid options match the attributes of ExecutionOptions.
    """
    global execution_options

    for key, value in kwargs.items():
        if hasattr(execution_options, key):
            setattr(execution_options, key, value)
        else:
            raise ValueError(f"Invalid execution option: {key}")


def get_execution_options() -> ExecutionOptions:
    """Get a copy of the current execution options.

    Returns:
        A copy of the current execution options.
    """
    return dataclasses.replace(execution_options)
