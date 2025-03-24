"""
Execution options for XCS graph execution.

This module provides configuration options for graph execution, allowing
customization of dispatching strategy, parallelism, and device selection.
"""

import dataclasses
from typing import Any, Dict, Generator, Optional, TypeVar, cast

from ember.core.exceptions import InvalidArgumentError


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
    scheduler: Optional[
        str
    ] = None  # For compatibility with examples using scheduler="sequential"


# Global default execution options
_global_options = ExecutionOptions()


def set_execution_options(**kwargs: Any) -> None:
    """Updating the global execution options.

    Setting or modifying global execution options that affect how the XCS engine
    executes graphs and operations, including parallelism settings and
    device selection strategy.

    Args:
        **kwargs: Keyword arguments to update execution options.
            Valid options match the attributes of ExecutionOptions.

    Raises:
        ValueError: If an invalid option name is provided.
    """
    global _global_options

    for key, value in kwargs.items():
        if key == "scheduler" and value == "sequential":
            # Handle the legacy scheduler="sequential" option by setting use_parallel=False
            setattr(_global_options, "use_parallel", False)
            setattr(_global_options, key, value)
        elif hasattr(_global_options, key):
            setattr(_global_options, key, value)
        else:
            raise InvalidArgumentError.with_context(
                f"Invalid execution option: {key}",
                option=key,
                valid_options=list(dataclasses.asdict(_global_options).keys()),
            )


def get_execution_options() -> ExecutionOptions:
    """Getting a copy of the current execution options.

    Creating a copy of the current global execution options to prevent
    accidental modifications to the global state.

    Returns:
        A copy of the current execution options.
    """
    return dataclasses.replace(_global_options)


class _ExecutionOptionsContext:
    """Context manager for temporarily modifying execution options.

    Allowing temporarily setting execution options for a specific block of code
    without permanently changing the global settings. The original options
    are automatically restored after the context exits.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializing with desired execution options.

        Args:
            **kwargs: Execution options to set for the context. Valid options
                match the attributes of ExecutionOptions.
        """
        self.kwargs = kwargs
        self.original_options = None

    def __enter__(self) -> ExecutionOptions:
        """Entering the context by saving original options and applying new ones.

        Returns:
            The current global execution options instance with modified settings.
        """
        self.original_options = get_execution_options()
        set_execution_options(**self.kwargs)
        return _global_options

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exiting the context by restoring original execution options."""
        if self.original_options:
            # Restore the original options by directly updating the global instance
            for key, value in dataclasses.asdict(self.original_options).items():
                setattr(_global_options, key, value)


def execution_options(**kwargs: Any) -> _ExecutionOptionsContext:
    """Temporarily modifying execution options for a block of code.

    Creating a context manager that allows temporarily changing execution options
    within a specific scope. The original options are restored after the context exits.

    Args:
        **kwargs: Execution options to set for the context.
            Valid options match the attributes of ExecutionOptions.

    Returns:
        A context manager that applies the specified execution options.

    Example:
        ```python
        # Run with parallel execution and 4 workers
        with execution_options(use_parallel=True, max_workers=4):
            result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})

        # Run with sequential execution
        with execution_options(scheduler="sequential"):
            result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
        ```
    """
    return _ExecutionOptionsContext(**kwargs)
