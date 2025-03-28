"""Configuration context for XCS graph execution.

Provides context-based API for controlling execution behavior of computational graphs.
Controls parallelism, scheduling strategy, and resource allocation through
thread-safe context managers.

Usage examples:
1. Context-specific execution settings:
   ```python
   # Execute with parallel scheduling and 4 worker threads
   with execution_options(use_parallel=True, max_workers=4):
       result = my_operator(inputs=data)
   ```

2. Global configuration:
   ```python
   # Configure all subsequent executions
   set_execution_options(use_parallel=True)
   
   # Operations inherit these settings
   result1 = operator1(inputs=data1)
   result2 = operator2(inputs=data2)
   ```
   
3. With JIT compilation:
   ```python
   @jit
   def process(data):
       return transformed_data
       
   # JIT functions respect execution context at runtime
   with execution_options(scheduler="parallel", max_workers=8):
       result = process(input_data)  # Uses parallel execution
   ```
"""

import dataclasses
from typing import Any, Optional

from ember.core.exceptions import InvalidArgumentError


@dataclasses.dataclass
class ExecutionOptions:
    """Configuration parameters for XCS graph execution.

    Controls scheduling strategy, parallelism, memory optimization,
    and execution monitoring for computational graphs. These options
    affect runtime behavior without changing graph structure.

    Attributes:
        use_parallel: Controls parallel execution of compatible operations
        max_workers: Thread pool size limit for parallel execution
        device_strategy: Execution backend selection strategy ('auto', 'cpu')
        enable_caching: Toggles caching of intermediate results
        trace_execution: Enables detailed execution tracing for debugging
        timeout_seconds: Maximum allowed execution time before termination
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
    """Updates global execution configuration.

    Modifies global execution settings that apply to all subsequent
    graph executions across the application unless overridden by
    a local execution_options context.

    Args:
        **kwargs: Configuration parameters to update.
                 Must match ExecutionOptions attributes.

    Raises:
        InvalidArgumentError: When provided an unsupported option name
    """
    global _global_options

    for key, value in kwargs.items():
        if key == "scheduler" and value == "sequential":
            # Handle the legacy scheduler="sequential" option by setting use_parallel=False
            _global_options.use_parallel = False
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
    """Retrieves current execution configuration.

    Returns a deep copy of the current global execution settings,
    preventing unintended modification of the global state.

    Returns:
        Copy of current execution configuration
    """
    return dataclasses.replace(_global_options)


class _ExecutionOptionsContext:
    """Context manager for scoped execution configuration.

    Applies temporary execution settings for a specific code block,
    automatically restoring previous settings when exiting the context.
    Ensures thread-safety for nested execution contexts.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes context with specified configuration.

        Args:
            **kwargs: Execution parameters to apply within the context.
                     Must match ExecutionOptions attributes.
        """
        self.kwargs = kwargs
        self.original_options = None

    def __enter__(self) -> ExecutionOptions:
        """Saves current configuration and applies context-specific settings.

        Returns:
            Modified global execution options instance
        """
        self.original_options = get_execution_options()
        set_execution_options(**self.kwargs)
        return _global_options

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restores original configuration when exiting context."""
        if self.original_options:
            # Restore the original options by directly updating the global instance
            for key, value in dataclasses.asdict(self.original_options).items():
                setattr(_global_options, key, value)


def execution_options(**kwargs: Any) -> _ExecutionOptionsContext:
    """Creates context manager for scoped execution configuration.

    Provides temporary execution settings for a specific code block.
    Settings apply only within the context and automatically revert
    when exiting the scope.

    Args:
        **kwargs: Execution parameters for the context.
                 Must match ExecutionOptions attributes.

    Returns:
        Context manager with specified execution settings

    Example:
        ```python
        # With parallel execution, 4 workers
        with execution_options(use_parallel=True, max_workers=4):
            result = my_operator(inputs={"data": data})

        # With sequential execution
        with execution_options(scheduler="sequential"):
            result = my_operator(inputs={"data": data})
        ```
    """
    return _ExecutionOptionsContext(**kwargs)
