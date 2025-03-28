"""
Structured Logging for XCS Framework

This module extends Ember's logging system with structured context data,
enabling consistent contextual metadata across the XCS execution chain.
It provides performance insights, execution tracing, and error diagnostics
through thread-local context propagation.

Key features:
1. Thread-local context storage for metadata propagation
2. Standard contextual fields (node_id, operation_type, timestamp)
3. Performance timing with configurable thresholds and sampling
4. Context managers for preserving log context across boundaries
5. Structured formatters for machine-readable log output
6. Performance-optimized logging with minimal overhead
7. Centralized configuration for logging verbosity and sampling

Performance optimization features:
- Conditional execution based on log level to avoid expensive operations
- Sampling for high-volume events to reduce logging overhead
- In-memory buffering of logs for high-throughput scenarios
- Smart serialization that only occurs when logs will actually be emitted
- Global enable/disable flags for production deployment
"""

import functools
import logging
import random
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# Type variables for more precise typing
T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


# Global configuration for structured logging
class LoggingConfig:
    """Configuration settings for structured logging with performance controls.

    Centralizing logging configuration to enable consistent performance tuning
    across the codebase without requiring changes to individual logger calls.

    Attributes:
        enabled: Master switch to enable/disable structured logging
        sampling_rate: Fraction of operations to log (1.0 = all, 0.1 = 10%)
        default_threshold_ms: Default threshold for timing logs (0 = log all)
        include_context_for_level: Minimum level to include context data (DEBUG = always)
        buffer_size: Size of in-memory log buffer (0 = immediate processing)
        trace_all_operations: Whether to trace all operations or only slow ones
        max_context_data_size: Maximum size in bytes for context data
    """

    enabled: bool = True
    sampling_rate: float = 1.0
    default_threshold_ms: float = 0.0
    include_context_for_level: int = logging.DEBUG
    buffer_size: int = 0
    trace_all_operations: bool = True
    max_context_data_size: int = 10000

    # Performance mode flags
    high_performance_mode: bool = False  # When True, minimizes logging
    development_mode: bool = False  # When True, maximizes logging

    @classmethod
    def configure(cls, **kwargs):
        """Configuring logging settings from kwargs."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

        # Special handling for performance/development modes
        if cls.high_performance_mode:
            cls.enabled = False
            cls.sampling_rate = 0.01
            cls.default_threshold_ms = 100.0
            cls.trace_all_operations = False

        if cls.development_mode:
            cls.enabled = True
            cls.sampling_rate = 1.0
            cls.default_threshold_ms = 0.0
            cls.trace_all_operations = True


# Thread-local storage for context data with proper isolation
_thread_local = threading.local()


def should_log(level: int, sampling_rate: Optional[float] = None) -> bool:
    """Determining if an operation should be logged based on level and sampling.

    Checking both the configured sampling rate and whether the given level
    is enabled for logging, to avoid expensive logging operations when
    they won't be emitted.

    Args:
        level: Logging level (e.g., logging.DEBUG)
        sampling_rate: Optional override for the global sampling rate

    Returns:
        True if the operation should be logged, False otherwise
    """
    if not LoggingConfig.enabled:
        return False

    # Check if this level is enabled in any handlers
    root_logger = logging.getLogger()
    if not root_logger.isEnabledFor(level):
        return False

    # Apply sampling rate - ensure proper float comparison for testing with mocks
    effective_rate = (
        sampling_rate if sampling_rate is not None else LoggingConfig.sampling_rate
    )
    # Convert to float to handle MagicMock objects in tests
    float_rate = float(effective_rate) if effective_rate is not None else 1.0
    if float_rate < 1.0:
        return random.random() < float_rate

    return True


def _get_context() -> Dict[str, Any]:
    """Accessing the current thread-local context dictionary.

    Creating the thread-local context if it doesn't exist yet,
    ensuring proper isolation between threads.

    Returns:
        The current thread-local context dictionary.
    """
    if not hasattr(_thread_local, "context"):
        _thread_local.context = {}
    return _thread_local.context


def get_context_value(key: str, default: T = None) -> Union[Any, T]:
    """Retrieving a value from the current logging context.

    Looking up a key in the thread-local context with a fallback
    default if the key doesn't exist.

    Args:
        key: Context key to retrieve.
        default: Default value if key doesn't exist.

    Returns:
        The value associated with the key or the default.
    """
    return _get_context().get(key, default)


def set_context_value(key: str, value: Any) -> None:
    """Setting a value in the current logging context.

    Storing a key-value pair in the thread-local context for
    later inclusion in log messages.

    Args:
        key: Context key to set.
        value: Value to store in context.
    """
    _get_context()[key] = value


def clear_context() -> None:
    """Clearing all values from the current logging context.

    Removing all context data from the thread-local storage,
    typically used when entering a new execution scope.
    """
    _get_context().clear()


@contextmanager
def log_context(**kwargs: Any) -> None:
    """Maintaining logging context within a code block.

    Preserving the previous context state and restoring it when
    exiting the block, ensuring proper context boundaries.

    Args:
        **kwargs: Key-value pairs to add to the context.

    Yields:
        None
    """
    # Creating a deep copy to ensure complete isolation
    original_context = _get_context().copy()
    try:
        # Updating context with new values
        _get_context().update(kwargs)
        yield
    finally:
        # Restoring original context atomically
        _thread_local.context = original_context


def with_context(
    logger: logging.Logger, level: int, msg: str, *args: Any, **kwargs: Any
) -> None:
    """Logging a message with the current context data included.

    Adding thread-local context data to the log record, ensuring
    consistent contextual metadata across the execution chain.
    Performance optimizations minimize overhead in hot paths.

    Args:
        logger: Logger instance to use.
        level: Logging level (e.g., logging.INFO).
        msg: Log message format string.
        *args: Message formatting args.
        **kwargs: Additional log data (will be merged with context).
    """
    # Fast path: skip all processing if logging is disabled or level won't be logged
    if not LoggingConfig.enabled or not logger.isEnabledFor(level):
        return

    # Performance optimization: only process context data for higher log levels
    if level >= LoggingConfig.include_context_for_level:
        # Creating a copy to avoid modifying the original kwargs
        log_data = kwargs.copy()

        # Adding thread-local context data
        context_data = _get_context()
        if context_data:
            # Performance optimization: limit context data size
            if LoggingConfig.max_context_data_size > 0:
                # Only copy context if we're truncating to avoid unnecessary copy
                if (
                    sys.getsizeof(str(context_data))
                    > LoggingConfig.max_context_data_size
                ):
                    context_copy = {}
                    total_size = 0
                    # Add items until we reach size limit
                    for k, v in context_data.items():
                        v_str = str(v)
                        size = sys.getsizeof(v_str)
                        if total_size + size <= LoggingConfig.max_context_data_size:
                            context_copy[k] = v
                            total_size += size
                        else:
                            # Indicate truncation
                            context_copy["_truncated"] = True
                            break
                    log_data["context"] = context_copy
                else:
                    # No truncation needed, use the original
                    log_data["context"] = context_data.copy()
            else:
                # No size limit, use the full context
                log_data["context"] = context_data.copy()

        # Adding timestamp if not present - use monotonic time for performance
        if "timestamp" not in log_data:
            log_data["timestamp"] = time.perf_counter()

        # Logging with combined data
        logger.log(level, msg, *args, extra={"structured_data": log_data})
    else:
        # For lower levels, skip context processing to improve performance
        logger.log(level, msg, *args)


def get_logger(name: str) -> logging.Logger:
    """Getting a logger with the XCS namespace.

    Creating a properly namespaced logger for XCS components,
    ensuring consistent logger naming throughout the codebase.

    Args:
        name: Logger name (will be prefixed with 'ember.xcs.' if not already).

    Returns:
        Configured logger instance.
    """
    if not name.startswith("ember.xcs."):
        name = f"ember.xcs.{name}"
    return logging.getLogger(name)


def time_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    threshold_ms: Optional[float] = None,
    sample_rate: Optional[float] = None,
) -> Callable[[F], F]:
    """Timing an operation and logging its duration.

    Decorating functions to measure and report execution time,
    with configurable thresholds and sampling for high-volume operations.
    The decorator includes performance optimizations to minimize overhead
    in production environments.

    Args:
        operation_name: Name of the operation being timed.
        logger: Logger to use (defaults to "ember.xcs.performance").
        level: Logging level for the timing message.
        threshold_ms: Only log if duration exceeds this threshold (milliseconds).
                      If None, uses LoggingConfig.default_threshold_ms.
        sample_rate: Fraction of calls to log (1.0 = log all, 0.1 = log 10%).
                    If None, uses LoggingConfig.sampling_rate.

    Returns:
        Decorated function that logs timing information with minimal overhead.

    Example:
        ```python
        @time_operation("graph_execution", threshold_ms=100)
        def execute_graph(graph, inputs):
            # Function implementation
            return result
        ```
    """

    def decorator(func: F) -> F:
        # Early optimization: If logging is globally disabled and we're in high performance
        # mode, return the original function without any wrapper to eliminate all overhead
        if not LoggingConfig.enabled and LoggingConfig.high_performance_mode:
            return func

        # Determining which logger to use (done once at decoration time)
        effective_logger = logger if logger is not None else get_logger("performance")

        # Resolve threshold setting once (at decoration time)
        effective_threshold = (
            threshold_ms
            if threshold_ms is not None
            else LoggingConfig.default_threshold_ms
        )

        # Resolve sampling rate once (at decoration time)
        effective_sample_rate = (
            sample_rate if sample_rate is not None else LoggingConfig.sampling_rate
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Fast path: check if we should log this operation, return immediately to original
            # function if not, avoiding all overhead of timing and context setup
            if not should_log(level, effective_sample_rate):
                return func(*args, **kwargs)

            # Only generate ID and context if we'll actually log
            operation_id = str(uuid.uuid4())
            node_id = get_context_value("node_id", "unknown")

            # Performance optimization: only use context manager if we're actually logging
            # to avoid the overhead of stack frame setup/teardown
            if LoggingConfig.trace_all_operations:
                context_manager = log_context(
                    operation_name=operation_name,
                    operation_id=operation_id,
                    node_id=node_id,
                    function=func.__qualname__,
                )
            else:
                context_manager = _null_context()  # No-op context manager

            with context_manager:
                # Always measure time - minimal overhead
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    # Only construct and emit log message if above threshold
                    # Use explicit float conversion to ensure comparison works with MagicMock objects in tests
                    float_duration_ms = float(duration_ms)
                    float_threshold = (
                        float(effective_threshold)
                        if effective_threshold is not None
                        else 0.0
                    )

                    if float_duration_ms >= float_threshold:
                        # Check again if level is enabled to avoid expensive message formatting
                        if effective_logger.isEnabledFor(level):
                            with_context(
                                effective_logger,
                                level,
                                "%s completed in %.2fms",
                                operation_name,
                                duration_ms,
                                duration_ms=duration_ms,
                                function=func.__qualname__,
                            )

        return cast(F, wrapper)

    return decorator


@contextmanager
def _null_context() -> None:
    """A no-op context manager for performance optimization.

    Providing a null implementation that avoids the overhead of
    context manager setup/teardown when logging is disabled.

    Yields:
        None
    """
    yield


class StructuredLogAdapter(logging.LoggerAdapter):
    """Adapting standard loggers to include structured context data.

    Automatically adding thread-local context to all log records,
    simplifying the use of structured logging throughout the codebase.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Processing the logging message and keyword arguments.

        Adding thread-local context and standard metadata to all log records
        passing through this adapter.

        Args:
            msg: Log message.
            kwargs: Logging keyword arguments.

        Returns:
            Tuple of (modified message, modified kwargs).
        """
        # Creating a copy to avoid modifying the original kwargs
        kwargs_copy = kwargs.copy()

        # Initializing structured_data dict if not present
        if "extra" not in kwargs_copy:
            kwargs_copy["extra"] = {}
        if "structured_data" not in kwargs_copy["extra"]:
            kwargs_copy["extra"]["structured_data"] = {}

        # Adding current context to structured_data
        context_data = _get_context()
        if context_data:
            kwargs_copy["extra"]["structured_data"]["context"] = context_data.copy()

        # Adding timestamp if not present
        if "timestamp" not in kwargs_copy["extra"]["structured_data"]:
            kwargs_copy["extra"]["structured_data"]["timestamp"] = time.time()

        return msg, kwargs_copy


def get_structured_logger(name: str) -> StructuredLogAdapter:
    """Getting a structured logger that automatically includes context data.

    Creating a logger adapter that adds thread-local context and standard
    metadata to all log messages.

    Args:
        name: Logger name (will be prefixed with 'ember.xcs.' if not already).

    Returns:
        StructuredLogAdapter instance wrapping the logger.
    """
    logger = get_logger(name)
    return StructuredLogAdapter(logger, {})


class JsonLogFormatter(logging.Formatter):
    """Formatting log records as JSON objects with structured data.

    Converting log records to machine-readable JSON format with
    standard fields and any structured context data.

    This formatter enables log aggregation and analysis by external tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formatting a log record as a JSON string.

        Converting the log record and any associated structured data
        into a consistent JSON format.

        Args:
            record: Log record to format.

        Returns:
            JSON-formatted log string.
        """
        import json

        # Adding basic log record data
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Adding file/line information for traceability
        if hasattr(record, "pathname") and record.pathname:
            log_data["file"] = record.pathname
            log_data["line"] = record.lineno

        # Adding thread information for concurrency analysis
        log_data["thread_id"] = record.thread
        log_data["thread_name"] = record.threadName

        # Adding structured data if present
        if hasattr(record, "structured_data") and record.structured_data:
            # Using .update() to merge at the top level instead of nesting
            log_data.update(record.structured_data)

        # Adding exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Common log context keys for consistency across the codebase
class LogContextKeys:
    """Standard keys for structured logging context.

    Defining common keys for context data to ensure consistent
    field names across all log messages.
    """

    NODE_ID = "node_id"
    OPERATION_ID = "operation_id"
    OPERATION_NAME = "operation_name"
    TIMESTAMP = "timestamp"
    DURATION_MS = "duration_ms"
    FUNCTION = "function"
    GRAPH_ID = "graph_id"
    BATCH_SIZE = "batch_size"
    ERROR = "error"


def configure_logging(
    *,
    environment: str = "development",
    enabled: Optional[bool] = None,
    sampling_rate: Optional[float] = None,
    threshold_ms: Optional[float] = None,
    trace_all: Optional[bool] = None,
    max_context_size: Optional[int] = None,
) -> None:
    """Configuring structured logging based on environment and explicit settings.

    Setting up logging configuration with appropriate defaults for different
    environments while allowing explicit overrides for specific parameters.

    Args:
        environment: "development", "production", or "performance"
        enabled: Whether structured logging is enabled
        sampling_rate: Fraction of operations to log (0.0 - 1.0)
        threshold_ms: Minimum duration threshold for timing logs in milliseconds
        trace_all: Whether to trace all operations or only those above threshold
        max_context_size: Maximum size in bytes for context data

    Example:
        ```python
        # Minimal overhead for production
        configure_logging(environment="production")

        # Maximum visibility for development
        configure_logging(environment="development")

        # Custom configuration
        configure_logging(
            environment="production",
            sampling_rate=0.05,
            threshold_ms=100
        )
        ```
    """
    config_kwargs = {}

    # Apply environment presets
    if environment == "development":
        config_kwargs["development_mode"] = True
        config_kwargs["high_performance_mode"] = False
    elif environment == "production":
        config_kwargs["development_mode"] = False
        config_kwargs["enabled"] = True
        config_kwargs["sampling_rate"] = 0.1
        config_kwargs["default_threshold_ms"] = 50.0
        config_kwargs["trace_all_operations"] = False
    elif environment == "performance":
        config_kwargs["high_performance_mode"] = True
        config_kwargs["development_mode"] = False

    # Apply explicit overrides
    if enabled is not None:
        config_kwargs["enabled"] = enabled
    if sampling_rate is not None:
        config_kwargs["sampling_rate"] = sampling_rate
    if threshold_ms is not None:
        config_kwargs["default_threshold_ms"] = threshold_ms
    if trace_all is not None:
        config_kwargs["trace_all_operations"] = trace_all
    if max_context_size is not None:
        config_kwargs["max_context_data_size"] = max_context_size

    # Apply configuration
    LoggingConfig.configure(**config_kwargs)


def enrich_exception(exception: Exception, **context: Any) -> Exception:
    """Enriching an exception with structured diagnostic context.

    Adding context data to exceptions that support it, or creating a wrapper
    exception with context when the original doesn't support enrichment.

    Args:
        exception: The original exception to enrich
        **context: Key-value pairs to add as diagnostic context

    Returns:
        The enriched exception (either the original or a wrapper)

    Example:
        ```python
        try:
            process_data()
        except ValueError as e:
            # Add diagnostic context before re-raising
            raise enrich_exception(e,
                operation="data_processing",
                input_size=len(data),
                timestamp=time.time()
            )
        ```
    """
    # Add current logging context if available - do this first so explicit context takes precedence
    current_context = _get_context()
    combined_context = {}
    if current_context:
        combined_context.update(current_context)

    # Add explicit context (will override any values from thread-local context)
    combined_context.update(context)

    # If exception already supports context, add directly
    if hasattr(exception, "add_context") and callable(exception.add_context):
        exception.add_context(**combined_context)
        return exception

    # For Ember exceptions without explicit context support, add as attributes
    if hasattr(exception, "__module__") and getattr(
        exception, "__module__", ""
    ).startswith("ember."):
        for k, v in combined_context.items():
            if not hasattr(exception, k):
                setattr(exception, k, v)
    # For standard exceptions that don't have context support, add as attributes
    elif not hasattr(exception, "add_context"):
        for (
            k,
            v,
        ) in (
            context.items()
        ):  # Only add explicitly provided context, not thread-local context
            if not hasattr(exception, k):
                setattr(exception, k, v)

    # No need to wrap, just return the enriched exception
    return exception
