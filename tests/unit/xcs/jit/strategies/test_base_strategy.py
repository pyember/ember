"""Unit tests for the base JIT compilation strategy.

Tests the core functionality of the Strategy protocol and BaseStrategy class,
ensuring consistent interfaces for analyzing and compiling functions.
"""

import inspect
from typing import Any, Dict

import pytest

from ember.xcs.jit.strategies.base_strategy import BaseStrategy, JITFallbackMixin
from ember.xcs.jit.cache import JITCache


class TestBaseStrategy:
    """Test suite for the BaseStrategy class."""
    
    def test_extract_common_features(self) -> None:
        """Test extraction of common features from functions."""
        # Create a strategy instance
        strategy = BaseStrategy()
        
        # Test with a simple function
        def simple_function(x: int) -> int:
            return x * 2
        
        features = strategy._extract_common_features(simple_function)
        assert features["is_function"] is True
        assert features["is_class"] is False
        assert features["is_method"] is False
        assert features["has_call"] is True
        assert features["has_forward"] is False
        assert features["has_specification"] is False
        assert features["module"] == __name__
        assert features["name"] == "simple_function"
        assert features["has_source"] is True
        assert features["source_lines"] > 0
        
        # Test with a class
        class TestClass:
            def __call__(self, x: int) -> int:
                return x * 2
                
            def forward(self, x: int) -> int:
                return x * 3
        
        features = strategy._extract_common_features(TestClass)
        assert features["is_function"] is False
        assert features["is_class"] is True
        assert features["is_method"] is False
        assert features["has_call"] is True  # Class with __call__ method should be detected
        assert features["has_forward"] is True  # Class with forward method should be detected
        
        # Test with class instance
        instance = TestClass()
        features = strategy._extract_common_features(instance)
        assert features["is_function"] is False
        assert features["is_class"] is False
        assert features["is_method"] is False
        assert features["has_call"] is True  # Instance has __call__
        assert features["has_forward"] is True  # Instance has forward
    
    def test_get_cache(self) -> None:
        """Test cache retrieval logic."""
        strategy = BaseStrategy()
        
        # Test with default cache
        default_cache = strategy._get_cache()
        assert isinstance(default_cache, JITCache)
        
        # Test with explicit cache
        explicit_cache = JITCache()
        retrieved = strategy._get_cache(explicit_cache)
        assert retrieved is explicit_cache
    
    def test_add_control_methods(self) -> None:
        """Test addition of control methods to compiled functions."""
        strategy = BaseStrategy()
        
        # Simple function and wrapper
        def original(x: int) -> int:
            return x * 2
            
        def compiled(x: int) -> int:
            return original(x)
        
        # Add control methods
        cache = JITCache()
        strategy._add_control_methods(compiled, original, cache)
        
        # Verify methods were added
        assert hasattr(compiled, "disable_jit")
        assert hasattr(compiled, "enable_jit")
        assert hasattr(compiled, "get_stats")
        assert hasattr(compiled, "_original_function")
        assert compiled._original_function is original
        
        # Test control method behavior
        compiled.disable_jit()
        assert getattr(compiled, "_jit_disabled", False) is True
        
        compiled.enable_jit()
        assert getattr(compiled, "_jit_disabled", False) is False
        
        # Stats should be available but empty for a new cache
        stats = compiled.get_stats()
        assert isinstance(stats, dict)


class TestJITFallbackMixin:
    """Test suite for the JITFallbackMixin."""
    
    def test_fallback_behavior(self) -> None:
        """Test fallback functionality."""
        # Create a concrete strategy with the mixin
        class TestStrategy(JITFallbackMixin, BaseStrategy):
            def should_fallback(self, func):
                # Fall back for functions with "fallback" in the name
                return "fallback" in func.__name__
        
        strategy = TestStrategy()
        
        # Function that should use fallback
        def test_fallback_func(x: int) -> int:
            return x * 2
            
        # Function that should not use fallback
        def test_normal_func(x: int) -> int:
            return x * 3
        
        # Check fallback decisions
        assert strategy.should_fallback(test_fallback_func) is True
        assert strategy.should_fallback(test_normal_func) is False
        
        # Test fallback compilation - should return the original function
        assert strategy.fallback_compile(test_fallback_func) is test_fallback_func