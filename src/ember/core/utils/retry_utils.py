"""
Resilient execution facilities with configurable retry policies.

This module provides a type-safe, composable framework for handling transient failures
through configurable retry strategies. It encapsulates backoff algorithms and retry
policies behind a clean interface, enabling resilient execution of arbitrary operations.

Key components:
- IRetryStrategy: Core abstraction for retry policy implementations
- ExponentialBackoffStrategy: Production-ready implementation using randomized exponential backoff
- run_with_backoff: Convenience function for common retry scenarios
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ImportError as err:
    raise ImportError(
        "tenacity is required for retry_utils. Install via Poetry: poetry add tenacity"
    ) from err

T = TypeVar("T")


class IRetryStrategy(ABC, Generic[T]):
    """Abstract base class for retry strategies.

    Defines the core contract for retry policy implementations, enabling
    interchangeable strategies with consistent execution semantics. Implementations
    determine specific backoff algorithms, retry conditions, and exception handling.

    This abstraction supports the Strategy pattern, allowing runtime selection of
    different retry behaviors while maintaining a consistent interface.
    """

    @abstractmethod
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a callable under a configured retry policy.

        Args:
            func: A callable representing the operation to perform.
            *args: Variable length argument list to pass to the callable.
            **kwargs: Arbitrary keyword arguments to pass to the callable.

        Returns:
            The result of the callable execution.

        Raises:
            Exception: Propagates the last exception encountered if all retry attempts fail.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ExponentialBackoffStrategy(IRetryStrategy[T]):
    """Retry strategy using randomized exponential backoff.

    Implements a jittered exponential backoff algorithm to mitigate the thundering
    herd problem in distributed systems. Wait times increase exponentially between
    retries with random variation, providing optimal retry behavior for network
    operations and distributed services.

    Attributes:
        min_wait: Minimum wait time (in seconds) before the first retry.
        max_wait: Maximum wait time (in seconds) allowed between retries.
        max_attempts: Total number of allowed retry attempts.
    """

    def __init__(
        self, min_wait: int = 1, max_wait: int = 60, max_attempts: int = 3
    ) -> None:
        """Initialize the exponential backoff strategy.

        Args:
            min_wait: Minimum wait time in seconds before retrying.
            max_wait: Maximum wait time in seconds between retries.
            max_attempts: Number of retry attempts before failing.
        """
        self.min_wait: int = min_wait
        self.max_wait: int = max_wait
        self.max_attempts: int = max_attempts

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute the callable with exponential backoff retries.

        Wraps the target function with tenacity's retry decorator, configuring
        randomized exponential backoff and maximum attempt constraints. The wrapped
        function propagates the original function's return type for strong typing.

        Args:
            func: The callable to execute with retry protection.
            *args: Positional arguments passed to the callable.
            **kwargs: Keyword arguments passed to the callable.

        Returns:
            The result returned by the callable upon successful execution.

        Raises:
            Exception: The last exception raised if all retries fail.
        """

        # Creating a wrapped function with tenacity retry configuration
        @retry(
            wait=wait_random_exponential(min=self.min_wait, max=self.max_wait),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        def wrapped() -> T:
            return func(*args, **kwargs)

        return wrapped()


# Singleton instance with default configuration for common use cases
_default_strategy: ExponentialBackoffStrategy[Any] = ExponentialBackoffStrategy()


def run_with_backoff(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a function with resilient retry behavior.

    Provides a simple interface for adding retry protection to any operation,
    particularly useful for network calls, database operations, and other
    potentially unstable external interactions.

    Args:
        func: The function to execute with retry protection.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the successful function execution.

    Example:
        ```python
        # Resilient API call with automatic retry
        response = run_with_backoff(
            requests.get,
            "https://api.example.com/data",
            headers={"Authorization": "Bearer token"}
        )

        # Resilient database operation
        results = run_with_backoff(
            db.execute_query,
            "SELECT * FROM users WHERE active = true",
            timeout=30
        )
        ```
    """
    return _default_strategy.execute(func, *args, **kwargs)


if __name__ == "__main__":

    def flaky_function(x: int) -> int:
        """Simulate a flaky function with random failures.

        Demonstrates retry behavior by randomly failing approximately
        half the time, simulating transient infrastructure issues.

        Args:
            x: An integer input for the computation.

        Returns:
            The computed result (x multiplied by 2).

        Raises:
            RuntimeError: On simulated random failure.
        """
        import random

        if random.random() < 0.5:
            raise RuntimeError("Simulated transient failure!")
        return x * 2

    # Implementation for demonstration purposes
