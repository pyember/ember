"""Unit tests for JIT performance metrics.

This module tests the collection and reporting of JIT performance metrics,
including compilation time, execution time, and cache statistics.
"""

import time
from typing import Any, Dict

import pytest

from ember.xcs.tracer.tracer_decorator import JITMetrics, JITCache, _jit_cache


def test_jit_metrics_basic() -> None:
    """Test basic recording and reporting of JIT metrics."""
    # Create a new metrics instance
    metrics = JITMetrics()

    # Record some metrics
    metrics.record_compilation(0.1)
    metrics.record_execution(0.2)
    metrics.record_tracing(0.05)

    metrics.record_cache_hit()
    metrics.record_cache_hit()
    metrics.record_cache_miss()

    # Check the recorded values
    assert metrics.compilation_time == 0.1
    assert metrics.execution_time == 0.2
    assert metrics.tracing_time == 0.05

    assert metrics.cache_hits == 2
    assert metrics.cache_misses == 1

    # Check the calculated values
    assert metrics.cache_hit_ratio == 2 / 3

    # Test the reset functionality
    metrics.reset()
    assert metrics.compilation_time == 0.0
    assert metrics.execution_time == 0.0
    assert metrics.tracing_time == 0.0
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 0


def test_jit_cache_with_metrics() -> None:
    """Test that JITCache correctly records metrics."""
    # Create a new cache
    cache = JITCache[str]()

    # Reset metrics to ensure clean start
    cache.reset_metrics()

    # Create test objects for keys
    class TestKey:
        def __init__(self, name):
            self.name = name

    key1 = TestKey("key1")
    key2 = TestKey("key2")
    non_existent_key = TestKey("non_existent")

    # Test cache miss
    result = cache.get_with_state(non_existent_key)
    assert result is None
    assert cache.metrics.cache_misses == 1
    assert cache.metrics.cache_hits == 0

    # Test cache hit
    cache.set(key1, "test_value")
    result = cache.get_with_state(key1)
    assert result == "test_value"
    assert cache.metrics.cache_hits == 1
    assert cache.metrics.cache_misses == 1

    # Test state invalidation
    cache.set(key2, "state_value", "state1")
    result = cache.get_with_state(key2, "state1")
    assert result == "state_value"
    assert cache.metrics.cache_hits == 2

    # State changed, should be a miss
    result = cache.get_with_state(key2, "state2")
    assert result is None
    assert cache.metrics.cache_misses == 2

    # Test metrics copying
    metrics_copy = cache.get_metrics()
    assert metrics_copy.cache_hits == 2
    assert metrics_copy.cache_misses == 2

    # Modifying the copy doesn't affect the original
    metrics_copy.cache_hits = 100
    assert cache.metrics.cache_hits == 2


@pytest.mark.parametrize(
    "compilation_time,execution_time,cache_hits,cache_misses,expected_hit_ratio",
    [
        (0.1, 0.2, 10, 0, 1.0),
        (0.1, 0.2, 0, 10, 0.0),
        (0.1, 0.2, 7, 3, 0.7),
    ],
)
def test_jit_metrics_parameterized(
    compilation_time: float,
    execution_time: float,
    cache_hits: int,
    cache_misses: int,
    expected_hit_ratio: float,
) -> None:
    """Test metrics calculation with different parameters."""
    metrics = JITMetrics()

    metrics.record_compilation(compilation_time)
    metrics.record_execution(execution_time)

    for _ in range(cache_hits):
        metrics.record_cache_hit()

    for _ in range(cache_misses):
        metrics.record_cache_miss()

    assert metrics.compilation_time == compilation_time
    assert metrics.execution_time == execution_time
    assert metrics.cache_hits == cache_hits
    assert metrics.cache_misses == cache_misses
    assert metrics.cache_hit_ratio == expected_hit_ratio


def test_global_jit_cache() -> None:
    """Test that the global JIT cache instance works correctly."""
    # Reset metrics to ensure clean start
    _jit_cache.reset_metrics()

    # Use string test keys instead of actual operators for simplicity
    class TestObj:
        pass

    obj1 = TestObj()
    obj2 = TestObj()

    # Create a test graph (using a string for simplicity)
    _jit_cache.set(obj1, "test_graph1")

    # First access should be a hit
    result = _jit_cache.get_with_state(obj1)
    assert result == "test_graph1"
    assert _jit_cache.metrics.cache_hits == 1

    # Non-existent key should be a miss
    result = _jit_cache.get_with_state(obj2)
    assert result is None
    assert _jit_cache.metrics.cache_misses == 1

    # Get metrics string representation
    metrics_str = str(_jit_cache.metrics)
    assert "JIT Performance Metrics:" in metrics_str
    assert "Cache hit ratio: 50.00%" in metrics_str
