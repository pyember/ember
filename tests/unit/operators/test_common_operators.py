"""Tests for built-in operators."""

from collections import Counter
from typing import Any

import pytest

from ember.operators import (
    Cache,
    Chain,
    Ensemble,
    Operator,
    Retry,
    Router,
    chain,
    ensemble,
    router,
)

# Global call tracker for testing
_call_tracker = {}


# Helper operators for testing
class MockOperator(Operator):
    """Mock operator that returns predefined result."""

    result: Any

    def __init__(self, result):
        self.result = result
        # Use id as unique identifier
        _call_tracker[id(self)] = 0

    def forward(self, input):
        _call_tracker[id(self)] += 1
        if callable(self.result):
            return self.result()
        return self.result

    @property
    def call_count(self):
        return _call_tracker.get(id(self), 0)


class TestEnsemble:
    """Test the Ensemble operator."""

    def test_ensemble_without_aggregator(self):
        """Test ensemble returns list when no aggregator."""
        op1 = MockOperator("result1")
        op2 = MockOperator("result2")
        op3 = MockOperator("result3")

        ens = Ensemble([op1, op2, op3])

        results = ens("input")

        # Should return list of all results
        assert results == ["result1", "result2", "result3"]

        # All operators called
        assert op1.call_count == 1
        assert op2.call_count == 1
        assert op3.call_count == 1

    def test_ensemble_with_aggregator(self):
        """Test ensemble with custom aggregator function."""
        op1 = MockOperator("A")
        op2 = MockOperator("B")
        op3 = MockOperator("A")

        # Majority vote aggregator
        def majority_vote(results):
            counts = Counter(results)
            return counts.most_common(1)[0][0]

        ens = Ensemble([op1, op2, op3], aggregator=majority_vote)

        result = ens("input")

        # Should return most common result
        assert result == "A"

    def test_ensemble_numeric_aggregator(self):
        """Test ensemble with numeric aggregation."""
        op1 = MockOperator(10)
        op2 = MockOperator(20)
        op3 = MockOperator(30)

        # Average aggregator
        def average(results):
            return sum(results) / len(results)

        ens = Ensemble([op1, op2, op3], aggregator=average)

        result = ens("input")
        assert result == 20.0

    def test_ensemble_convenience_function(self):
        """Test ensemble() convenience function."""
        op1 = MockOperator("A")
        op2 = MockOperator("B")

        # Using convenience function
        ens = ensemble(op1, op2)

        results = ens("input")
        assert results == ["A", "B"]


class TestChain:
    """Test the Chain operator."""

    def test_chain_sequential_execution(self):
        """Test chain passes output sequentially."""

        # Operators that transform their input
        class UpperOperator(Operator):
            def forward(self, text):
                return text.upper()

        class AddSuffixOperator(Operator):
            def forward(self, text):
                return text + " WORLD"

        class ExclamationOperator(Operator):
            def forward(self, text):
                return text + "!"

        ch = Chain([UpperOperator(), AddSuffixOperator(), ExclamationOperator()])

        result = ch("hello")

        # Should transform: "hello" -> "HELLO" -> "HELLO WORLD" -> "HELLO WORLD!"
        assert result == "HELLO WORLD!"

    def test_chain_numeric_operations(self):
        """Test chain with numeric operations."""

        class AddOne(Operator):
            def forward(self, x):
                return x + 1

        class Double(Operator):
            def forward(self, x):
                return x * 2

        ch = Chain([AddOne(), Double(), AddOne()])

        result = ch(5)
        # 5 -> 6 -> 12 -> 13
        assert result == 13

    def test_chain_convenience_function(self):
        """Test chain() convenience function."""

        class Increment(Operator):
            def forward(self, x):
                return x + 1

        ch = chain(Increment(), Increment(), Increment())

        result = ch(0)
        assert result == 3


class TestRouter:
    """Test the Router operator."""

    def test_router_basic_routing(self):
        """Test router directs to correct operator."""

        # Route by input type
        def get_route(x):
            if isinstance(x, str):
                return "string"
            elif isinstance(x, int):
                return "number"
            else:
                return "other"

        routes = {
            "string": MockOperator("processed string"),
            "number": MockOperator("processed number"),
            "other": MockOperator("processed other"),
        }

        rt = Router(routes, get_route)

        assert rt("hello") == "processed string"
        assert rt(42) == "processed number"
        assert rt([1, 2, 3]) == "processed other"

    def test_router_with_default(self):
        """Test router with default route."""

        def get_route(x):
            if x > 0:
                return "positive"
            else:
                return "invalid"  # Not in routes

        routes = {"positive": MockOperator("positive result")}

        rt = Router(routes, get_route, default_route="positive")

        # Normal routing
        assert rt(5) == "positive result"

        # Falls back to default
        assert rt(-5) == "positive result"

    def test_router_missing_route_error(self):
        """Test router raises error for missing route."""
        routes = {"exists": MockOperator("result")}

        rt = Router(routes, lambda x: "missing")

        with pytest.raises(KeyError) as exc_info:
            rt("input")

        assert "No operator for route 'missing'" in str(exc_info.value)

    def test_router_convenience_function(self):
        """Test router() convenience function."""
        routes = {"a": MockOperator("result_a"), "b": MockOperator("result_b")}

        rt = router(routes, router_fn=lambda x: x)

        assert rt("a") == "result_a"
        assert rt("b") == "result_b"


class TestRetry:
    """Test the Retry operator."""

    def test_retry_success_first_try(self):
        """Test retry when operation succeeds immediately."""
        op = MockOperator("success")
        retry_op = Retry(op)

        result = retry_op("input")

        assert result == "success"
        assert op.call_count == 1

    def test_retry_eventual_success(self):
        """Test retry succeeds after failures."""
        call_count = 0

        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure {call_count}")
            return "success"

        op = MockOperator(failing_then_success)
        retry_op = Retry(op, max_attempts=3)

        result = retry_op("input")

        assert result == "success"
        assert call_count == 3

    def test_retry_max_attempts_exceeded(self):
        """Test retry fails after max attempts."""

        def always_fail():
            raise Exception("Always fails")

        op = MockOperator(always_fail)
        retry_op = Retry(op, max_attempts=3)

        with pytest.raises(Exception) as exc_info:
            retry_op("input")

        assert "Always fails" in str(exc_info.value)
        assert op.call_count == 3

    def test_retry_custom_should_retry(self):
        """Test retry with custom retry logic."""

        class CustomError(Exception):
            pass

        def should_retry_custom(e, attempt):
            # Only retry CustomError, not other exceptions
            return isinstance(e, CustomError) and attempt < 3

        def raise_custom():
            raise CustomError("Custom error")

        op = MockOperator(raise_custom)
        retry_op = Retry(op, should_retry=should_retry_custom)

        with pytest.raises(CustomError):
            retry_op("input")

        # Should have tried 3 times
        assert op.call_count == 3


class TestCache:
    """Test the Cache operator."""

    def test_cache_basic_caching(self):
        """Test basic caching behavior."""
        call_count = 0

        class CountingOperator(Operator):
            def forward(self, x):
                nonlocal call_count
                call_count += 1
                return x * 2

        op = CountingOperator()
        cached_op = Cache(op)

        # First call
        result1 = cached_op(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same input - should use cache
        result2 = cached_op(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different input - should call operator
        result3 = cached_op(3)
        assert result3 == 6
        assert call_count == 2

    def test_cache_key_function(self):
        """Test cache with custom key function."""

        class DictOperator(Operator):
            def forward(self, data):
                return data["value"] * 2

        # Key only on 'value' field
        def get_key(data):
            return str(data.get("value"))

        op = DictOperator()
        cached_op = Cache(op, key_fn=get_key)

        # These should use same cache entry
        result1 = cached_op({"value": 5, "other": "a"})
        result2 = cached_op({"value": 5, "other": "b"})

        assert result1 == 10
        assert result2 == 10

    def test_cache_max_size(self):
        """Test cache respects max size limit."""
        call_history = []

        class IdentityOperator(Operator):
            def forward(self, x):
                call_history.append(x)
                return x

        op = IdentityOperator()
        cached_op = Cache(op, max_size=2)

        # Fill cache beyond limit
        cached_op(1)  # Cache: {1}
        cached_op(2)  # Cache: {1, 2}
        cached_op(3)  # Cache: {2, 3} - evicts 1

        # This should call operator again (1 was evicted)
        cached_op(1)

        assert call_history == [1, 2, 3, 1]
