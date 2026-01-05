"""Robust error handling patterns for AI applications.

This example demonstrates:
- Exception handling for LLM operations
- Retry strategies with exponential backoff
- Graceful degradation patterns
- Error aggregation for batch operations

Run with:
    python examples/07_error_handling/robust_patterns.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ember.api import op

T = TypeVar("T")


# =============================================================================
# Part 1: Basic Error Handling
# =============================================================================

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class RateLimitError(ModelError):
    """Raised when rate limits are exceeded."""
    pass


class TimeoutError(ModelError):
    """Raised when a request times out."""
    pass


class InvalidResponseError(ModelError):
    """Raised when model response is invalid."""
    pass


@op
def call_with_basic_handling(prompt: str) -> Dict[str, Any]:
    """Demonstrate basic try/except patterns."""
    try:
        # Simulate model call
        if "error" in prompt.lower():
            raise ModelError("Simulated model error")

        return {
            "success": True,
            "response": f"Response to: {prompt[:30]}",
        }
    except ModelError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def demonstrate_basic_handling() -> None:
    """Show basic error handling."""
    print("Part 1: Basic Error Handling")
    print("-" * 50)

    # Successful call
    result = call_with_basic_handling("What is Python?")
    print(f"Success case: {result}")

    # Error case
    result = call_with_basic_handling("This will error")
    print(f"Error case: {result}")
    print()


# =============================================================================
# Part 2: Retry Strategies
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (RateLimitError, TimeoutError)


def retry_with_backoff(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
) -> T:
    """Execute function with exponential backoff retry."""
    config = config or RetryConfig()
    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            return func()
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt + 1 == config.max_attempts:
                break

            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay,
            )

            # Add jitter to prevent thundering herd
            if config.jitter:
                delay *= 0.5 + random.random()

            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)

    raise last_exception or Exception("Retry failed")


def demonstrate_retry() -> None:
    """Show retry with exponential backoff."""
    print("Part 2: Retry Strategies")
    print("-" * 50)

    call_count = 0

    def flaky_operation() -> str:
        """Simulate a flaky operation that succeeds on third try."""
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            raise RateLimitError("Rate limit exceeded")
        return "Success after retries!"

    config = RetryConfig(
        max_attempts=5,
        base_delay=0.1,  # Short delays for demo
        jitter=False,
    )

    try:
        result = retry_with_backoff(flaky_operation, config)
        print(f"Result: {result}")
        print(f"Total attempts: {call_count}")
    except Exception as e:
        print(f"Failed after all retries: {e}")
    print()


# =============================================================================
# Part 3: Result Types for Error Handling
# =============================================================================

@dataclass
class Result:
    """Result type that captures success or failure."""

    success: bool
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, value: Any, **metadata: Any) -> "Result":
        """Create a successful result."""
        return cls(success=True, value=value, metadata=metadata)

    @classmethod
    def fail(cls, error: Exception, **metadata: Any) -> "Result":
        """Create a failure result."""
        return cls(
            success=False,
            error=str(error),
            error_type=type(error).__name__,
            metadata=metadata,
        )

    def unwrap(self) -> Any:
        """Get value or raise if failed."""
        if self.success:
            return self.value
        raise ModelError(f"{self.error_type}: {self.error}")

    def unwrap_or(self, default: Any) -> Any:
        """Get value or return default if failed."""
        return self.value if self.success else default


@op
def safe_operation(value: int) -> Result:
    """Operation that returns Result instead of raising."""
    if value < 0:
        return Result.fail(ValueError("Value must be non-negative"))
    if value > 100:
        return Result.fail(ValueError("Value must be <= 100"))

    return Result.ok(value * 2, original=value)


def demonstrate_result_types() -> None:
    """Show Result type pattern."""
    print("Part 3: Result Types")
    print("-" * 50)

    # Success case
    result = safe_operation(42)
    print(f"Success: value={result.unwrap()}, metadata={result.metadata}")

    # Error case with safe access
    result = safe_operation(-5)
    print(f"Error: {result.error}, fallback={result.unwrap_or('default')}")

    # Multiple operations
    print("\nBatch with mixed results:")
    for val in [10, -1, 50, 150, 25]:
        result = safe_operation(val)
        status = "OK" if result.success else f"ERR: {result.error}"
        print(f"  {val:4} -> {status}")
    print()


# =============================================================================
# Part 4: Graceful Degradation
# =============================================================================

@dataclass
class FallbackChain:
    """Chain of fallback operations."""

    operations: List[Callable[[], Any]]
    names: List[str]

    def execute(self) -> Dict[str, Any]:
        """Try operations in order until one succeeds."""
        errors = []

        for op_func, name in zip(self.operations, self.names):
            try:
                result = op_func()
                return {
                    "success": True,
                    "value": result,
                    "source": name,
                    "fallback_count": len(errors),
                }
            except Exception as e:
                errors.append({"source": name, "error": str(e)})

        return {
            "success": False,
            "errors": errors,
            "message": "All fallbacks exhausted",
        }


def demonstrate_graceful_degradation() -> None:
    """Show graceful degradation patterns."""
    print("Part 4: Graceful Degradation")
    print("-" * 50)

    def primary_model():
        raise RateLimitError("Primary model overloaded")

    def secondary_model():
        raise TimeoutError("Secondary model timeout")

    def fallback_model():
        return "Response from fallback model"

    chain = FallbackChain(
        operations=[primary_model, secondary_model, fallback_model],
        names=["gpt-4", "gpt-3.5-turbo", "local-model"],
    )

    result = chain.execute()
    print(f"Source: {result.get('source', 'none')}")
    print(f"Fallback count: {result.get('fallback_count', 0)}")
    print(f"Value: {result.get('value', result.get('message'))}")
    print()


# =============================================================================
# Part 5: Batch Error Aggregation
# =============================================================================

@dataclass
class BatchResult:
    """Aggregated results from batch processing."""

    results: List[Result]

    @property
    def successes(self) -> List[Result]:
        return [r for r in self.results if r.success]

    @property
    def failures(self) -> List[Result]:
        return [r for r in self.results if not r.success]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successes) / len(self.results)

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.results),
            "successes": len(self.successes),
            "failures": len(self.failures),
            "success_rate": f"{self.success_rate:.1%}",
            "error_types": self._count_error_types(),
        }

    def _count_error_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.failures:
            error_type = r.error_type or "Unknown"
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts


def process_batch_with_errors(items: List[int]) -> BatchResult:
    """Process batch, collecting all results."""
    results = [safe_operation(item) for item in items]
    return BatchResult(results=results)


def demonstrate_batch_aggregation() -> None:
    """Show batch error aggregation."""
    print("Part 5: Batch Error Aggregation")
    print("-" * 50)

    items = [10, -5, 50, 200, 25, -10, 75, 150]
    batch_result = process_batch_with_errors(items)

    summary = batch_result.summary()
    print(f"Summary: {summary}")

    print("\nSuccessful values:")
    for r in batch_result.successes:
        print(f"  {r.metadata.get('original')} -> {r.value}")

    print("\nFailed items:")
    for r in batch_result.failures:
        print(f"  Error: {r.error}")
    print()


# =============================================================================
# Part 6: Circuit Breaker Pattern
# =============================================================================

@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    failure_threshold: int = 5
    reset_timeout: float = 30.0
    _failure_count: int = 0
    _last_failure_time: float = 0
    _state: str = "closed"  # closed, open, half-open

    def call(self, func: Callable[[], T]) -> T:
        """Execute function through circuit breaker."""
        if self._state == "open":
            if time.time() - self._last_failure_time > self.reset_timeout:
                self._state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func()
            if self._state == "half-open":
                self._state = "closed"
                self._failure_count = 0
            return result
        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = "open"

            raise e

    @property
    def state(self) -> str:
        return self._state


def demonstrate_circuit_breaker() -> None:
    """Show circuit breaker pattern."""
    print("Part 6: Circuit Breaker Pattern")
    print("-" * 50)

    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)

    def always_fails():
        raise Exception("Service unavailable")

    # Trip the breaker
    for i in range(5):
        try:
            breaker.call(always_fails)
        except Exception as e:
            print(f"  Call {i + 1}: {e} (state={breaker.state})")

    print(f"\nFinal state: {breaker.state}")
    print("(Further calls will fail fast until reset timeout)")
    print()


def main() -> None:
    """Demonstrate robust error handling patterns."""
    print("Robust Error Handling Patterns")
    print("=" * 50)
    print()

    demonstrate_basic_handling()
    demonstrate_retry()
    demonstrate_result_types()
    demonstrate_graceful_degradation()
    demonstrate_batch_aggregation()
    demonstrate_circuit_breaker()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use specific exception types for different error conditions")
    print("2. Implement retry with exponential backoff and jitter")
    print("3. Consider Result types for explicit error handling")
    print("4. Build fallback chains for graceful degradation")
    print("5. Aggregate errors in batch operations for visibility")
    print("6. Use circuit breakers to prevent cascading failures")
    print()
    print("Next: Explore examples/08_advanced_patterns/ for advanced usage")


if __name__ == "__main__":
    main()
