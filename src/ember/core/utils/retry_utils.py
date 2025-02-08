from __future__ import annotations
from typing import Any, Callable, Generic, TypeVar
from abc import ABC, abstractmethod

try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ImportError as err:
    raise ImportError(
        "tenacity is required for retry_utils. Install via pip install tenacity"
    ) from err

T = TypeVar("T")


class IRetryStrategy(ABC, Generic[T]):
    """Abstract base class for retry strategies.

    Implementations of this interface define how to execute a callable function
    with retries and backoff policies.
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
    """Retry strategy using tenacity's exponential backoff.

    This strategy waits for a random, exponentially increasing amount of time
    between retries, with configurable minimum and maximum wait times and a
    maximum number of retry attempts.

    Attributes:
        min_wait (int): Minimum wait time (in seconds) before the first retry.
        max_wait (int): Maximum wait time (in seconds) allowed between retries.
        max_attempts (int): Total number of allowed retry attempts.
    """

    def __init__(
        self, min_wait: int = 1, max_wait: int = 60, max_attempts: int = 3
    ) -> None:
        """Initialize the exponential backoff strategy.

        Args:
            min_wait: Minimum wait time in seconds before retrying.
            max_wait: Maximum wait time in seconds for waiting between retries.
            max_attempts: Number of retry attempts before the operation is considered failed.
        """
        self.min_wait: int = min_wait
        self.max_wait: int = max_wait
        self.max_attempts: int = max_attempts

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute the given callable using exponential backoff for retries.

        The function is wrapped with tenacity's retry decorator using a random
        exponential wait time between retries and stops after a defined number
        of attempts. Exceptions are re-raised when the retry policy is exhausted.

        Args:
            func: The callable to execute.
            *args: Positional arguments passed to the callable.
            **kwargs: Keyword arguments passed to the callable.

        Returns:
            The result returned by the callable.

        Raises:
            Exception: The last exception raised by the callable if all retries fail.
        """

        @retry(
            wait=wait_random_exponential(min=self.min_wait, max=self.max_wait),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        def wrapped() -> T:
            return func(*args, **kwargs)

        return wrapped()


_default_strategy: ExponentialBackoffStrategy[Any] = ExponentialBackoffStrategy()


def run_with_backoff(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a callable with a default exponential backoff strategy.

    This convenience wrapper leverages a default ExponentialBackoffStrategy to
    execute the provided function. It is useful for operations that may experience
    transient failures, such as network calls.

    Args:
        func: The callable to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the callable execution.

    Example:
        result = run_with_backoff(some_network_call, "hello", param=123)
    """
    return _default_strategy.execute(func, *args, **kwargs)


if __name__ == "__main__":

    def flaky_function(x: int) -> int:
        """Simulate a flaky function that randomly fails to mimic transient errors.

        Args:
            x: An integer input used in the computation.

        Returns:
            The computed result, which is x multiplied by 2.

        Raises:
            RuntimeError: If a simulated transient error occurs.
        """
        import random

        if random.random() < 0.5:
            raise RuntimeError("Simulated failure!")
        return x * 2

    print("Demo: Executing a flaky function with backoff (up to 3 attempts).")
    try:
        result: int = run_with_backoff(flaky_function, 10)
        print(f"Success, output is: {result}")
    except Exception as exc:
        print(f"Failed after retries: {exc}")
