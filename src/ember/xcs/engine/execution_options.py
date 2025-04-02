"""Configuration context for XCS graph execution.

Provides a thread-safe, context-based API for controlling computational graph execution.
Manages parallelism, scheduling strategy, and resource allocation through
immutable configuration objects and thread-local storage.

Usage:
1. Context-specific execution:
   ```python
   with execution_options(scheduler="parallel", max_workers=4):
       result = my_operator(inputs=data)
   ```

2. Global configuration:
   ```python
   set_execution_options(scheduler="parallel")
   
   # Operations inherit settings
   result1 = operator1(inputs=data1)
   result2 = operator2(inputs=data2)
   ```
   
3. With JIT compilation:
   ```python
   @jit
   def process(data):
       return transformed_data
       
   with execution_options(scheduler="parallel", max_workers=8):
       result = process(input_data)  # Uses parallel execution
   ```
"""

import dataclasses
import threading
from typing import Any, Dict, Literal, Optional, Set, Union, FrozenSet

from ember.core.exceptions import InvalidArgumentError

# =============================================================================
# Execution configuration constants
# =============================================================================

# Valid scheduler strategies
SchedulerType = Literal["sequential", "parallel", "wave", "auto", "noop"]
VALID_SCHEDULER_TYPES: FrozenSet[str] = frozenset(
    ["sequential", "parallel", "wave", "auto", "noop"]
)

# Valid executor types
VALID_EXECUTORS: FrozenSet[str] = frozenset(
    ["auto", "async", "thread"]
)

# =============================================================================
# Option validation utilities
# =============================================================================

def validate_option(
    option_name: str, 
    value: str, 
    valid_values: FrozenSet[str]
) -> None:
    """Validate that an option value is allowed.
    
    Args:
        option_name: Name of the option being validated
        value: The option value
        valid_values: Set of valid values
        
    Raises:
        InvalidArgumentError: If value is not valid
    """
    if value.lower() not in valid_values:
        raise InvalidArgumentError.with_context(
            f"Invalid {option_name}: '{value}'. Valid values: {', '.join(sorted(valid_values))}",
            context={
                "option": option_name,
                "value": value,
                "valid_values": sorted(valid_values)
            }
        )


@dataclasses.dataclass(frozen=True)
class ExecutionOptions:
    """Immutable configuration for XCS graph execution.

    Controls scheduling, parallelism, and optimization of computational graphs.
    Thread-safe through immutability.

    Attributes:
        use_parallel: Controls parallel execution of operations
        max_workers: Thread pool size for parallel execution
        device_strategy: Execution backend selection ('auto', 'cpu')
        enable_caching: Toggles intermediate result caching
        trace_execution: Enables detailed execution tracing
        timeout_seconds: Maximum execution time before termination
        collect_metrics: Whether to collect detailed performance metrics
        debug: Whether to enable debug output
        scheduler: Scheduler strategy or instance (overrides use_parallel)
            Valid values: "sequential", "parallel", "wave", "auto", "noop"
        
        # Execution options
        executor: Executor selection mode
            Valid values: "auto", "async", "thread"
        fail_fast: If True, fails immediately on errors
                 If False, continues execution after errors
    """

    # Original options preserved for backward compatibility
    use_parallel: bool = True
    max_workers: Optional[int] = None
    device_strategy: str = "auto"
    enable_caching: bool = True
    trace_execution: bool = False
    timeout_seconds: Optional[float] = None
    collect_metrics: bool = False
    debug: bool = False
    # Scheduler can be:
    # - String with strategy name: "sequential", "parallel", "auto", "noop", "wave"
    # - BaseScheduler instance for direct control
    # - None to use default based on use_parallel
    scheduler: Optional[Union[SchedulerType, Any]] = None
    
    # Execution options with simplified naming
    executor: str = "auto"
    fail_fast: bool = True

    def __post_init__(self) -> None:
        """Validates configuration values."""
        # Validate max_workers
        if (self.max_workers is not None and 
                (not isinstance(self.max_workers, int) or self.max_workers <= 0)):
            raise InvalidArgumentError.with_context(
                "max_workers must be a positive integer or None",
                max_workers=self.max_workers,
            )

        # Validate scheduler
        if (isinstance(self.scheduler, str) and 
                self.scheduler.lower() not in _SCHEDULER_MAP):
            raise InvalidArgumentError.with_context(
                f"Unknown scheduler: {self.scheduler}",
                scheduler=self.scheduler,
                valid_schedulers=list(_SCHEDULER_MAP.keys()),
            )
            
        # Validate executor type
        if isinstance(self.executor, str):
            validate_option("executor", self.executor, VALID_EXECUTORS)
            
        # fail_fast is a boolean, no validation needed


# Scheduler name to parallel setting mapping
_SCHEDULER_MAP: Dict[str, bool] = {
    "sequential": False,
    "parallel": True,
    "auto": True,
    "noop": False,
    "wave": True,
}

# Thread-local storage
_LOCAL = threading.local()

# Global options and lock
_GLOBAL_OPTIONS = ExecutionOptions()
_GLOBAL_LOCK = threading.RLock()


def get_execution_options() -> ExecutionOptions:
    """Retrieves current execution configuration.

    Returns thread-local options if set, otherwise global options.
    Never returns a reference to shared state.

    Returns:
        Current execution configuration (immutable)
    """
    return getattr(_LOCAL, "options", _GLOBAL_OPTIONS)


def set_execution_options(**kwargs: Any) -> ExecutionOptions:
    """Updates global execution configuration atomically.

    Creates a new immutable configuration that applies to all subsequent
    graph executions unless overridden by a context.

    Args:
        **kwargs: Configuration parameters to update.
                 Must match ExecutionOptions attributes.

    Returns:
        New global execution options

    Raises:
        InvalidArgumentError: When provided invalid option or value
    """
    global _GLOBAL_OPTIONS
    
    # Validate option names against dataclass fields
    fields: Set[str] = {f.name for f in dataclasses.fields(ExecutionOptions)}
    invalid_keys = set(kwargs.keys()) - fields
    if invalid_keys:
        raise InvalidArgumentError.with_context(
            f"Invalid execution option(s): {', '.join(invalid_keys)}",
            invalid_options=sorted(invalid_keys),
            valid_options=sorted(fields),
        )
    
    # Handle scheduler-to-parallel mapping for backward compatibility
    if "scheduler" in kwargs and isinstance(kwargs["scheduler"], str):
        scheduler_name = kwargs["scheduler"].lower()
        
        # Only set use_parallel if not already in kwargs
        if "use_parallel" not in kwargs and scheduler_name in _SCHEDULER_MAP:
            kwargs["use_parallel"] = _SCHEDULER_MAP[scheduler_name]
    
    # Create new immutable options in a thread-safe manner
    with _GLOBAL_LOCK:
        current = dataclasses.asdict(_GLOBAL_OPTIONS)
        updated = {**current, **kwargs}
        new_options = ExecutionOptions(**updated)
        _GLOBAL_OPTIONS = new_options
        
    return new_options


def reset_execution_options() -> ExecutionOptions:
    """Resets execution options to system defaults.
    
    Clears thread-local options and resets global options.
    
    Returns:
        Default execution options
    """
    global _GLOBAL_OPTIONS
    
    # Clear thread-local options if set
    if hasattr(_LOCAL, "options"):
        delattr(_LOCAL, "options")
    
    # Reset global options atomically
    with _GLOBAL_LOCK:
        _GLOBAL_OPTIONS = ExecutionOptions()
        
    return _GLOBAL_OPTIONS


class _ExecutionContext:
    """Thread-safe context manager for scoped execution configuration.

    Applies temporary settings within a code block using thread-local storage,
    automatically restoring previous settings when exiting.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes context with validated configuration.

        Args:
            **kwargs: Execution parameters to apply within context.
                    Must match ExecutionOptions attributes.

        Raises:
            InvalidArgumentError: When provided invalid options/values
        """
        # Validate options early
        fields: Set[str] = {f.name for f in dataclasses.fields(ExecutionOptions)}
        invalid_keys = set(kwargs.keys()) - fields
        if invalid_keys:
            raise InvalidArgumentError.with_context(
                f"Invalid execution option(s): {', '.join(invalid_keys)}",
                invalid_options=sorted(invalid_keys),
                valid_options=sorted(fields)
            )
            
        # Handle scheduler-to-parallel mapping for backward compatibility
        if "scheduler" in kwargs and isinstance(kwargs["scheduler"], str):
            scheduler_name = kwargs["scheduler"].lower()
            
            # Only set use_parallel if not explicitly provided
            if "use_parallel" not in kwargs and scheduler_name in _SCHEDULER_MAP:
                kwargs["use_parallel"] = _SCHEDULER_MAP[scheduler_name]
                
        self.kwargs = kwargs
        self.previous = None

    def __enter__(self) -> ExecutionOptions:
        """Sets thread-local options for the context duration.

        Returns:
            New execution options for this context
        """
        # Save previous thread-local options if they exist
        self.previous = getattr(_LOCAL, "options", None)
        
        # Create new options based on current settings
        current = dataclasses.asdict(get_execution_options())
        updated = {**current, **self.kwargs}
        
        # Set thread-local options
        _LOCAL.options = ExecutionOptions(**updated)
        
        return _LOCAL.options

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restores previous thread-local options on exit."""
        if self.previous is not None:
            # Restore previous thread-local options
            _LOCAL.options = self.previous
        else:
            # Clear thread-local options
            if hasattr(_LOCAL, "options"):
                delattr(_LOCAL, "options")


def execution_options(**kwargs: Any) -> _ExecutionContext:
    """Creates a thread-safe context for scoped execution settings.

    Settings apply only within the context and automatically revert
    when exiting. Thread-local storage ensures contexts are properly
    isolated between threads.

    Args:
        **kwargs: Execution parameters for the context.
                Must match ExecutionOptions attributes.

    Returns:
        Context manager with specified settings

    Raises:
        InvalidArgumentError: If options/values are invalid
        
    Example:
        ```python
        # Parallel execution with 4 workers
        with execution_options(scheduler="parallel", max_workers=4):
            result = my_operator(inputs=data)

        # Sequential execution
        with execution_options(scheduler="sequential"):
            result = process(data)
        ```
    """
    return _ExecutionContext(**kwargs)