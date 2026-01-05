"""Tests for built-in operators."""

import threading
from collections import Counter
from typing import Any, Dict, List, Optional

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


# Global thread-safe call tracker for tests
class CallTracker:
    """Thread-safe call tracker for test assertions."""

    def __init__(self):
        self._calls: Dict[str, List[Any]] = {}
        self._lock = threading.Lock()

    def track(self, operator_id: str, input_val: Any):
        """Track a call."""
        with self._lock:
            if operator_id not in self._calls:
                self._calls[operator_id] = []
            self._calls[operator_id].append(input_val)

    def get_count(self, operator_id: str) -> int:
        """Get call count for operator."""
        with self._lock:
            return len(self._calls.get(operator_id, []))

    def reset(self):
        """Reset all tracking."""
        with self._lock:
            self._calls.clear()


# Singleton tracker for tests
tracker = CallTracker()


class MockOperator(Operator):
    """Mock operator that returns predefined result - Ember Module compatible."""

    result: Any
    operator_id: str

    def __init__(self, result: Any, operator_id: Optional[str] = None):
        """Initialize with result."""
        self.result = result
        self.operator_id = operator_id or f"mock_{id(self)}"

    def forward(self, input: Any) -> Any:
        """Execute the mock operation."""
        tracker.track(self.operator_id, input)

        if callable(self.result):
            return self.result()
        return self.result


class TestEnsemble:
    """Test the Ensemble operator."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()

    @pytest.mark.parametrize(
        "results,expected",
        [
            pytest.param(
                ["result1", "result2", "result3"], ["result1", "result2", "result3"], id="strings"
            ),
            pytest.param([1, 2, 3], [1, 2, 3], id="numbers"),
            pytest.param([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}], id="dicts"),
            pytest.param([], [], id="empty"),
        ],
    )
    def test_ensemble_without_aggregator(self, results, expected):
        """Test ensemble returns list when no aggregator."""
        operators = [MockOperator(r, f"op_{i}") for i, r in enumerate(results)]
        ens = Ensemble(operators)

        actual = ens("input")
        assert actual == expected

        # All operators called exactly once
        for i, _operator in enumerate(operators):
            assert tracker.get_count(f"op_{i}") == 1

    @pytest.mark.parametrize(
        "results,aggregator,expected",
        [
            pytest.param(
                ["A", "B", "A"], lambda x: Counter(x).most_common(1)[0][0], "A", id="majority-vote"
            ),
            pytest.param([10, 20, 30], lambda x: sum(x) / len(x), 20.0, id="average"),
            pytest.param([10, 20, 30], max, 30, id="max"),
            pytest.param(
                ["hello", "world"], lambda x: " ".join(x), "hello world", id="join-strings"
            ),
        ],
    )
    def test_ensemble_with_aggregator(self, results, aggregator, expected):
        """Test ensemble with various aggregators."""
        operators = [MockOperator(r, f"op_{i}") for i, r in enumerate(results)]
        ens = Ensemble(operators, aggregator=aggregator)

        actual = ens("input")
        assert actual == expected

    def test_ensemble_exception_handling(self):
        """Test ensemble handles operator exceptions."""
        op1 = MockOperator("good", "op1")
        op2 = MockOperator(lambda: 1 / 0, "op2")  # Will raise exception
        op3 = MockOperator("also good", "op3")

        ens = Ensemble([op1, op2, op3])

        with pytest.raises(ZeroDivisionError):
            ens("input")

        # First operator called, third might not be due to exception
        assert tracker.get_count("op1") == 1

    def test_ensemble_convenience_function(self):
        """Test ensemble() convenience function."""
        op1 = MockOperator("A", "op1")
        op2 = MockOperator("B", "op2")

        # Using convenience function
        ens = ensemble(op1, op2)

        results = ens("input")
        assert results == ["A", "B"]


class TestChain:
    """Test the Chain operator."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()

    # Custom operators for chain testing
    class UpperOperator(Operator):
        name: str = "upper"

        def forward(self, text: str) -> str:
            return text.upper()

    class AddSuffixOperator(Operator):
        suffix: str
        name: str = "add_suffix"

        def __init__(self, suffix: str):
            self.suffix = suffix

        def forward(self, text: str) -> str:
            return text + self.suffix

    class MathOperator(Operator):
        operation_name: str
        name: str = "math"

        def __init__(self, operation_name: str):
            self.operation_name = operation_name

        def forward(self, x: float) -> float:
            if self.operation_name == "add1":
                return x + 1
            elif self.operation_name == "mul2":
                return x * 2
            elif self.operation_name == "sub3":
                return x - 3
            else:
                return x

    @pytest.mark.parametrize(
        "input_val,operators,expected",
        [
            pytest.param(
                "hello",
                lambda: [
                    TestChain.UpperOperator(),
                    TestChain.AddSuffixOperator(" WORLD"),
                    TestChain.AddSuffixOperator("!"),
                ],
                "HELLO WORLD!",
                id="string-chain",
            ),
            pytest.param(
                5,
                lambda: [
                    TestChain.MathOperator("add1"),
                    TestChain.MathOperator("mul2"),
                    TestChain.MathOperator("sub3"),
                ],
                9,  # (5 + 1) * 2 - 3 = 9
                id="numeric-chain",
            ),
        ],
    )
    def test_chain_operations(self, input_val, operators, expected):
        """Test chaining with various operations."""
        # Call operators lambda to get fresh instances
        ch = Chain(operators())
        result = ch(input_val)
        assert result == expected

    def test_empty_chain(self):
        """Test chain with no operators returns input."""
        ch = Chain([])
        assert ch("input") == "input"

    def test_single_operator_chain(self):
        """Test chain with single operator."""
        op = self.UpperOperator()
        ch = Chain([op])

        assert ch("hello") == "HELLO"

    def test_chain_convenience_function(self):
        """Test chain() convenience function."""
        increment = self.MathOperator("add1")
        ch = chain(increment, increment, increment)

        result = ch(0)
        assert result == 3


class TestRouter:
    """Test the Router operator."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()

    @pytest.mark.parametrize(
        "input_val,expected_route,expected_result",
        [
            pytest.param("hello", "string", "processed string", id="route-string"),
            pytest.param(42, "number", "processed number", id="route-number"),
            pytest.param([1, 2, 3], "other", "processed other", id="route-other"),
        ],
    )
    def test_router_basic_routing(self, input_val, expected_route, expected_result):
        """Test router directs to correct operator."""

        def get_route(x):
            if isinstance(x, str):
                return "string"
            elif isinstance(x, int):
                return "number"
            else:
                return "other"

        routes = {
            "string": MockOperator("processed string", "string_op"),
            "number": MockOperator("processed number", "number_op"),
            "other": MockOperator("processed other", "other_op"),
        }

        rt = Router(routes, get_route)
        result = rt(input_val)
        assert result == expected_result

    def test_router_with_default(self):
        """Test router with default route."""

        def get_route(x):
            if x > 0:
                return "positive"
            else:
                return "invalid"  # Not in routes

        routes = {"positive": MockOperator("positive result", "pos_op")}
        rt = Router(routes, get_route, default_route="positive")

        # Normal routing
        assert rt(5) == "positive result"

        # Falls back to default
        assert rt(-5) == "positive result"

    def test_router_missing_route_error(self):
        """Test router raises error for missing route."""
        routes = {"exists": MockOperator("result", "exists_op")}
        rt = Router(routes, lambda x: "missing")

        with pytest.raises(KeyError) as exc_info:
            rt("input")

        assert "missing" in str(exc_info.value)

    def test_router_convenience_function(self):
        """Test router() convenience function."""
        routes = {
            "a": MockOperator("result_a", "route_a"),
            "b": MockOperator("result_b", "route_b"),
        }
        rt = router(routes, router_fn=lambda x: x)

        assert rt("a") == "result_a"
        assert rt("b") == "result_b"


class TestRetry:
    """Test the Retry operator."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()

    @pytest.mark.parametrize(
        "failures_before_success,max_attempts,should_succeed",
        [
            pytest.param(0, 3, True, id="immediate-success"),
            pytest.param(2, 3, True, id="success-after-retries"),
            pytest.param(3, 3, False, id="exceeds-max-attempts"),
        ],
    )
    def test_retry_scenarios(self, failures_before_success, max_attempts, should_succeed):
        """Test various retry scenarios."""
        # Use a shared counter for attempt tracking
        attempt_counter = {"count": 0}

        def flaky_operation():
            attempt_counter["count"] += 1
            if attempt_counter["count"] <= failures_before_success:
                raise ValueError("Temporary failure")
            return "success"

        op = MockOperator(flaky_operation, "retry_op")
        retry_op = Retry(op, max_attempts=max_attempts)

        if should_succeed:
            result = retry_op("input")
            assert result == "success"
            assert tracker.get_count("retry_op") == failures_before_success + 1
        else:
            with pytest.raises(ValueError):
                retry_op("input")
            assert tracker.get_count("retry_op") == max_attempts

    def test_retry_custom_should_retry(self):
        """Test retry with custom retry logic."""

        class CustomError(Exception):
            pass

        def should_retry_custom(e, attempt):
            # Only retry CustomError, not other exceptions
            return isinstance(e, CustomError) and attempt < 3

        def raise_custom():
            raise CustomError("Custom error")

        op = MockOperator(raise_custom, "custom_retry_op")
        retry_op = Retry(op, should_retry=should_retry_custom)

        with pytest.raises(CustomError):
            retry_op("input")

        # Should have tried 3 times
        assert tracker.get_count("custom_retry_op") == 3


class TestCache:
    """Test the Cache operator."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()

    class CountingOperator(Operator):
        """Operator that counts calls via external tracker."""

        operation_name: str
        tracker_id: str

        def __init__(self, operation_name: str = "double", tracker_id: str = "counting_op"):
            self.operation_name = operation_name
            self.tracker_id = tracker_id

        def forward(self, x):
            tracker.track(self.tracker_id, x)
            if self.operation_name == "double":
                return x * 2
            elif self.operation_name == "format":
                return f"processed_{x}"
            else:
                return x

    def test_cache_basic_caching(self):
        """Test basic caching behavior."""
        op = self.CountingOperator(tracker_id="cache_test")
        cached_op = Cache(op)

        # First call
        result1 = cached_op(5)
        assert result1 == 10
        assert tracker.get_count("cache_test") == 1

        # Second call with same input - should use cache
        result2 = cached_op(5)
        assert result2 == 10
        assert tracker.get_count("cache_test") == 1  # Not incremented

        # Different input - should call operator
        result3 = cached_op(3)
        assert result3 == 6
        assert tracker.get_count("cache_test") == 2

    @pytest.mark.parametrize(
        "inputs,expected_calls",
        [
            pytest.param(["A", "A", "A"], 1, id="all-same"),
            pytest.param(["A", "B", "A", "B"], 2, id="two-unique"),
            pytest.param(["A", "B", "C"], 3, id="all-different"),
        ],
    )
    def test_cache_hit_rates(self, inputs, expected_calls):
        """Test cache with various input patterns."""
        op = self.CountingOperator("format", f"cache_hit_test_{id(inputs)}")
        cached_op = Cache(op)

        results = []
        for inp in inputs:
            results.append(cached_op(inp))

        # Verify results are correct
        expected_results = [f"processed_{x}" for x in inputs]
        assert results == expected_results

        # Verify cache worked
        assert tracker.get_count(f"cache_hit_test_{id(inputs)}") == expected_calls

    def test_cache_key_function(self):
        """Test cache with custom key function."""

        class DictOperator(Operator):
            name: str = "dict_op"

            def forward(self, data):
                tracker.track("dict_cache_op", data)
                return data["value"] * 2

        # Key only on 'value' field
        def get_key(data):
            return data["value"]

        op = DictOperator()
        cached_op = Cache(op, key_fn=get_key)

        # Same value, different metadata - should use cache
        result1 = cached_op({"value": 5, "meta": "a"})
        result2 = cached_op({"value": 5, "meta": "b"})

        assert result1 == 10
        assert result2 == 10
        assert tracker.get_count("dict_cache_op") == 1  # Only called once


# Contract tests for all operators
class TestOperatorContracts:
    """Verify all operators follow expected contracts."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()

    @pytest.mark.parametrize("operator_class", [Ensemble, Chain, Router, Retry, Cache])
    def test_operators_are_operators(self, operator_class):
        """Test all operator classes inherit from Operator."""
        assert issubclass(operator_class, Operator)

    def test_all_operators_have_forward(self):
        """Test all operators have forward method."""
        operators = [
            Ensemble([MockOperator(1)]),
            Chain([MockOperator(lambda x: x)]),
            Router({"A": MockOperator(1)}, lambda x: "A"),
            Retry(MockOperator(1)),
            Cache(MockOperator(1)),
        ]

        for op in operators:
            assert hasattr(op, "forward")
            assert callable(op.forward)

    def test_all_operators_callable(self):
        """Test all operators implement __call__."""
        operators = [
            Ensemble([MockOperator(1)]),
            Chain([MockOperator(lambda: "result")]),  # Fixed lambda
            Router({"A": MockOperator(1)}, lambda x: "A"),
            Retry(MockOperator(1)),
            Cache(MockOperator(1)),
        ]

        for op in operators:
            assert callable(op)
            # Should not raise
            result = op("test_input")
            assert result is not None


# Thread safety tests
class TestThreadSafety:
    """Test thread safety of operators."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker.reset()

    def test_ensemble_thread_safety(self):
        """Test ensemble is thread-safe."""
        results = []
        errors = []

        def make_operator(value):
            return MockOperator(lambda: value, f"thread_op_{value}")

        ens = Ensemble([make_operator(i) for i in range(10)])

        def run_ensemble(thread_id):
            try:
                result = ens(f"input_{thread_id}")
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(20):
            t = threading.Thread(target=run_ensemble, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20

        # All results should be lists of 10 elements
        for _thread_id, result in results:
            assert len(result) == 10
            assert result == list(range(10))

    def test_cache_thread_safety(self):
        """Test cache is thread-safe with concurrent access."""

        # Use a regular function instead of lambda with arg
        def double_func():
            # Cache will pass the input through, we need to handle it differently
            return lambda x: x * 2

        # Create a custom operator for this test
        class DoubleOperator(Operator):
            name: str = "double"

            def forward(self, x):
                tracker.track("thread_cache_op", x)
                return x * 2

        op = DoubleOperator()
        cached_op = Cache(op)

        results = []

        def access_cache(value):
            # Each thread accesses same values multiple times
            for _ in range(5):
                result = cached_op(value)
                results.append((value, result))

        threads = []
        # Multiple threads accessing same values
        for i in range(10):
            t = threading.Thread(target=access_cache, args=(i % 3,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all results are correct
        for value, result in results:
            assert result == value * 2

        # Only 3 unique values, so only 3 calls to underlying operator
        assert tracker.get_count("thread_cache_op") == 3
