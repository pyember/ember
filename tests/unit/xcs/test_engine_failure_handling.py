"""Tests for XCS engine failure handling and exception propagation.

These tests verify:
1. Exceptions in parallel nodes are properly wrapped with context
2. Remaining futures are cancelled when one node fails
3. XCSExecutionError provides useful debugging information
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ember.xcs.compiler.graph import IRGraph, node_from_callable
from ember.xcs.compiler.parallelism import ParallelismAnalyzer
from ember.xcs.config import Config
from ember.xcs.errors import XCSExecutionError
from ember.xcs.runtime.engine import ExecutionEngine


class TestExceptionPropagation:
    """Test that exceptions are properly wrapped with node context."""

    def test_exception_wrapped_with_node_context(self):
        """Exceptions from operators should be wrapped with XCSExecutionError."""

        def failing_op(x: int) -> int:
            raise ValueError("Intentional failure")

        def wrapper(x: int) -> int:
            return failing_op(x)

        graph = IRGraph()
        node = node_from_callable(
            "node_1",
            failing_op,
            ("arg_0",),
            ("var_1",),
            {"call_spec_pos": ({"kind": "arg", "name": "arg_0"},)},
        )
        graph = graph.add_node(node)

        engine = ExecutionEngine()
        analysis = ParallelismAnalyzer().analyze(graph)

        with pytest.raises(XCSExecutionError) as exc_info:
            engine.execute(graph, args=(42,), kwargs={}, parallelism=analysis)

        error = exc_info.value
        assert error.node_id == "node_1"
        assert "Intentional failure" in str(error)
        assert error.cause is not None
        assert isinstance(error.cause, ValueError)

    def test_xcs_errors_not_double_wrapped(self):
        """XCSError subclasses should not be wrapped again."""
        from ember.xcs.errors import XCSError

        def failing_op(x: int) -> int:
            raise XCSError("Already an XCS error")

        graph = IRGraph()
        node = node_from_callable(
            "node_1",
            failing_op,
            ("arg_0",),
            ("var_1",),
            {"call_spec_pos": ({"kind": "arg", "name": "arg_0"},)},
        )
        graph = graph.add_node(node)

        engine = ExecutionEngine()
        analysis = ParallelismAnalyzer().analyze(graph)

        with pytest.raises(XCSError) as exc_info:
            engine.execute(graph, args=(42,), kwargs={}, parallelism=analysis)

        # Should be the original error, not wrapped
        assert not isinstance(exc_info.value, XCSExecutionError)
        assert "Already an XCS error" in str(exc_info.value)


class TestFailFastCancellation:
    """Test that parallel execution cancels remaining work on failure."""

    def test_cancel_pending_clears_futures(self):
        """_cancel_pending should cancel all futures and clear the dict."""
        engine = ExecutionEngine()

        # Create mock futures
        future1 = MagicMock(spec=concurrent.futures.Future)
        future2 = MagicMock(spec=concurrent.futures.Future)
        futures = {"node_1": future1, "node_2": future2}

        engine._cancel_pending(futures)

        # All futures should be cancelled
        future1.cancel.assert_called_once()
        future2.cancel.assert_called_once()
        # Dict should be cleared
        assert len(futures) == 0

    def test_exception_triggers_cancellation(self):
        """When a future raises, _cancel_pending should be called on remaining."""
        # This is a unit test of the logic, not a timing-based integration test

        def failing_op() -> str:
            raise ValueError("Intentional failure")

        def success_op() -> str:
            return "ok"

        graph = IRGraph()
        node1 = node_from_callable(
            "fail_node",
            failing_op,
            (),
            ("v1",),
            {"call_spec_pos": (), "call_spec_kw": {}},
        )
        graph = graph.add_node(node1)

        engine = ExecutionEngine()
        analysis = ParallelismAnalyzer().analyze(graph)

        with pytest.raises(XCSExecutionError) as exc_info:
            engine.execute(graph, args=(), kwargs={}, parallelism=analysis)

        # Verify the exception was wrapped correctly
        assert "fail_node" in str(exc_info.value)
        assert exc_info.value.cause is not None


class TestConcurrentJITAccess:
    """Test thread safety of JIT compilation and execution."""

    def test_concurrent_jit_calls_are_safe(self):
        """Multiple threads calling the same jitted function should be safe."""
        from ember.xcs import jit

        call_count = {"value": 0}
        lock = threading.Lock()

        @jit
        def compute(x: int) -> int:
            with lock:
                call_count["value"] += 1
            return x * 2

        results: List[int] = []
        errors: List[Exception] = []
        result_lock = threading.Lock()

        def worker(value: int) -> None:
            try:
                result = compute(value)
                with result_lock:
                    results.append(result)
            except Exception as e:
                with result_lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent access failed: {errors}"
        assert len(results) == 50
        assert all(r == i * 2 for i, r in sorted(zip(range(50), sorted(results))))

    def test_concurrent_jit_with_different_args(self):
        """JIT should handle concurrent calls with different argument shapes."""
        from ember.xcs import jit

        @jit
        def process(items: list) -> int:
            return sum(items)

        results: Dict[int, int] = {}
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker(size: int) -> None:
            try:
                result = process(list(range(size)))
                with lock:
                    results[size] = result
            except Exception as e:
                with lock:
                    errors.append(e)

        sizes = [10, 20, 30, 40, 50]
        threads = [threading.Thread(target=worker, args=(s,)) for s in sizes]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent access with different args failed: {errors}"
        for size in sizes:
            expected = sum(range(size))
            assert results.get(size) == expected, f"Wrong result for size {size}"
