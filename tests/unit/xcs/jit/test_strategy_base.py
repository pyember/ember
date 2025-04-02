"""Tests for the base JIT strategy functionality."""

import inspect
import pytest
from typing import Dict, Any, Callable

from ember.xcs.jit.strategies.base_strategy import BaseStrategy, JITFallbackMixin


# Simple test class to create a BaseStrategy instance
class TestStrategy(BaseStrategy):
    """Test implementation of BaseStrategy for testing."""
    
    def analyze(self, func: Callable) -> Dict[str, Any]:
        """Test implementation of analyze."""
        features = self._extract_common_features(func)
        return {
            "score": 50,
            "rationale": "Test rationale",
            "features": features
        }
    
    def compile(self, func: Callable, **kwargs) -> Callable:
        """Test implementation of compile."""
        return func


class TestFallbackStrategy(BaseStrategy, JITFallbackMixin):
    """Test implementation of BaseStrategy with JITFallbackMixin."""
    
    def analyze(self, func: Callable) -> Dict[str, Any]:
        """Test implementation of analyze."""
        features = self._extract_common_features(func)
        return {
            "score": 50,
            "rationale": "Test rationale",
            "features": features
        }
    
    def compile(self, func: Callable, **kwargs) -> Callable:
        """Test implementation of compile."""
        if self.should_fallback(func):
            return self.fallback_compile(func)
        return func


# Test functions/classes for analysis
def simple_function(*, inputs):
    """Simple test function."""
    return {"result": inputs["value"] * 2}


class SimpleClass:
    """Simple test class."""
    
    def __init__(self):
        self.value = 10
    
    def __call__(self, *, inputs):
        return {"result": inputs["value"] * self.value}
    
    def forward(self, *, inputs):
        return {"result": inputs["value"] * self.value}


class OperatorWithSpecification:
    """Test class with specification attribute."""
    
    specification = {"type": "test"}
    
    def forward(self, *, inputs):
        return {"result": inputs["value"] * 3}


def test_extract_common_features():
    """Test extraction of common features from different function types."""
    strategy = TestStrategy()
    
    # Test with simple function
    features = strategy._extract_common_features(simple_function)
    assert features["is_function"] is True
    assert features["is_class"] is False
    assert features["is_method"] is False
    # In Python, all functions are callable, so has_call could be True
    assert features["has_forward"] is False
    assert features["has_specification"] is False
    
    # Test with class
    features = strategy._extract_common_features(SimpleClass)
    assert features["is_function"] is False
    assert features["is_class"] is True
    assert features["is_method"] is False
    assert features["has_call"] is True
    assert features["has_forward"] is True
    assert features["has_specification"] is False
    
    # Test with class instance
    instance = SimpleClass()
    features = strategy._extract_common_features(instance)
    assert features["is_function"] is False
    assert features["is_class"] is False
    assert features["is_method"] is False
    assert features["has_call"] is True
    assert features["has_forward"] is True
    assert features["has_specification"] is False
    
    # Test with operator that has specification
    features = strategy._extract_common_features(OperatorWithSpecification)
    assert features["is_function"] is False
    assert features["is_class"] is True
    assert features["has_specification"] is True


def test_get_cache():
    """Test getting a cache instance."""
    strategy = TestStrategy()
    
    # Test getting default cache
    cache = strategy._get_cache()
    assert cache is not None
    
    # Test with explicit cache
    from ember.xcs.jit.cache import JITCache
    explicit_cache = JITCache()
    result_cache = strategy._get_cache(explicit_cache)
    assert result_cache is explicit_cache


def test_add_control_methods():
    """Test adding control methods to compiled functions."""
    strategy = TestStrategy()
    
    # Create a function with control methods added
    def test_func(*, inputs):
        return {"result": inputs["value"] * 2}
    
    # Get cache to pass to _add_control_methods
    from ember.xcs.jit.cache import get_cache
    cache = get_cache()
    
    # Add control methods
    strategy._add_control_methods(test_func, simple_function, cache)
    
    # Check that control methods were added
    assert hasattr(test_func, "disable_jit")
    assert hasattr(test_func, "enable_jit")
    assert hasattr(test_func, "get_stats")
    assert hasattr(test_func, "_original_function")
    assert test_func._original_function is simple_function
    assert hasattr(test_func, "_jit_strategy")
    assert test_func._jit_strategy == "test"
    
    # Test enable/disable methods
    test_func.disable_jit()
    assert getattr(test_func, "_jit_disabled", False) is True
    
    test_func.enable_jit()
    assert getattr(test_func, "_jit_disabled", False) is False


def test_fallback_mixin():
    """Test the JITFallbackMixin functionality."""
    strategy = TestFallbackStrategy()
    
    # Test default should_fallback (returns False)
    assert strategy.should_fallback(simple_function) is False
    
    # Test fallback_compile (returns original function)
    result = strategy.fallback_compile(simple_function)
    assert result is simple_function
    
    # Override should_fallback to return True and test compile
    strategy.should_fallback = lambda func: True
    result = strategy.compile(simple_function)
    assert result is simple_function