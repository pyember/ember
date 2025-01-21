from __future__ import annotations
from typing import Any, Callable, Generic, TypeVar, Protocol, Union
from abc import ABC, abstractmethod

# You can use tenacity or any other library. We'll show tenacity by default:
try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ImportError:
    raise ImportError(
        "tenacity is required for retry_utils. Install via pip install tenacity"
    )

T = TypeVar("T")


##############################################################
# 1) Retry Strategy Interface (SOLID: Interface Segregation)
##############################################################

class IRetryStrategy(ABC, Generic[T]):
    """
    Abstract base for a retry strategy. 
    Implementers define how to run a function with retries/backoff.
    """

    @abstractmethod
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute 'func(*args, **kwargs)' under a retry policy.
        Returns the function's output or raises the last exception if retries fail.
        """
        pass


##############################################################
# 2) Tenacity-Based Exponential Backoff Strategy
##############################################################

class ExponentialBackoffStrategy(IRetryStrategy[T]):
    """
    A concrete strategy using tenacity's exponential backoff. 
    By default: 
      - random exponential wait from 1..60 sec
      - up to 3 attempts
    Adjust as needed or create a new subclass for your scenario.
    """

    def __init__(
        self,
        min_wait: int = 1,
        max_wait: int = 60,
        max_attempts: int = 3,
    ):
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.max_attempts = max_attempts

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Uses tenacity's retry logic with random exponential backoff.
        """
        @retry(
            wait=wait_random_exponential(min=self.min_wait, max=self.max_wait),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True
        )
        def wrapped() -> T:
            return func(*args, **kwargs)

        return wrapped()


##############################################################
# 3) Minimal Utility Function for Quick Use
##############################################################

_default_strategy = ExponentialBackoffStrategy[Any]()

def run_with_backoff(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    A convenience wrapper that uses a default ExponentialBackoffStrategy
    to retry 'func(*args, **kwargs)' upon failure.
    
    Example usage:
        result = run_with_backoff(some_network_call, "hello", param=123)
    """
    return _default_strategy.execute(func, *args, **kwargs)


##############################################################
# 4) Example Usage (If running this file directly)
##############################################################
if __name__ == "__main__":

    def flaky_function(x: int) -> int:
        """
        Example function that fails randomly, e.g. 
        simulating a network or embedding-service call.
        """
        import random
        if random.random() < 0.5:
            raise RuntimeError("Simulated failure!")
        return x * 2

    print("Demo: calling a flaky function with backoff (up to 3 attempts).")

    try:
        val = run_with_backoff(flaky_function, 10)
        print(f"Success, output is {val}")
    except Exception as e:
        print(f"Failed after retries: {e}")