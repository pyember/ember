"""Base class for JIT compilation strategies.

Defines the Strategy protocol that all JIT strategies must implement,
ensuring consistent interfaces for analyzing and compiling functions.
"""

import inspect
import time
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

from ember.xcs.jit.cache import JITCache, get_cache

# Define type variable for functions
F = TypeVar("F", bound=Callable)


class Strategy(Protocol):
    """Protocol defining the interface for JIT compilation strategies.
    
    All JIT compilation strategies must implement this protocol to ensure
    consistent interfaces for analysis and compilation.
    """
    
    def analyze(self, func: Callable) -> Dict[str, Any]:
        """Analyze a function to determine if this strategy is suitable.
        
        Args:
            func: Function to analyze
            
        Returns:
            Dictionary with analysis results including:
            - score: Suitability score (higher is better)
            - rationale: Reason for the score
        """
        ...
    
    def compile(self, 
                func: F, 
                sample_input: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> F:
        """Compile a function using this strategy.
        
        Args:
            func: Function to compile
            sample_input: Optional sample input for eager compilation
            **kwargs: Additional strategy-specific options
            
        Returns:
            Compiled function with the same signature
        """
        ...


class BaseStrategy:
    """Base class for JIT compilation strategies.
    
    Provides common functionality for all JIT strategies, including
    feature extraction, caching, and control methods.
    """
    
    def _extract_common_features(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """Extract common features from a function for analysis.
        
        Args:
            func: Function to analyze
            
        Returns:
            Dictionary with common features
        """
        # Basic features
        features = {
            "is_class": inspect.isclass(func),
            "is_function": inspect.isfunction(func),
            "is_method": inspect.ismethod(func),
            "has_call": hasattr(func, "__call__") and callable(getattr(func, "__call__")),
            "has_forward": hasattr(func, "forward") and callable(getattr(func, "forward")),
            "has_specification": hasattr(func, "specification"),
            "module": getattr(func, "__module__", "unknown"),
            "name": getattr(func, "__name__", "unnamed"),
        }
        
        # Try to get source code features
        try:
            source = inspect.getsource(func)
            features["has_source"] = True
            features["source_lines"] = len(source.splitlines())
            features["source_size"] = len(source)
        except (TypeError, OSError, IOError):
            features["has_source"] = False
            features["source_lines"] = 0
            features["source_size"] = 0
        
        return features
    
    def _get_cache(self, cache: Optional[JITCache] = None) -> JITCache:
        """Get the JIT cache to use.
        
        Args:
            cache: Optional explicit cache
            
        Returns:
            JIT cache instance
        """
        if cache is None:
            return get_cache()
        return cache
    
    def _add_control_methods(self, compiled_func: Callable, original_func: Callable, cache: JITCache) -> None:
        """Add control methods to the compiled function.
        
        Args:
            compiled_func: Compiled function
            original_func: Original function
            cache: JIT cache
        """
        # Add disable/enable methods
        def disable_jit() -> None:
            setattr(compiled_func, "_jit_disabled", True)
        
        def enable_jit() -> None:
            setattr(compiled_func, "_jit_disabled", False)
        
        def get_stats() -> Dict[str, Any]:
            return cache.get_metrics(original_func)
        
        compiled_func.disable_jit = disable_jit
        compiled_func.enable_jit = enable_jit
        compiled_func.get_stats = get_stats
        compiled_func._original_function = original_func
        compiled_func._jit_strategy = self.__class__.__name__.replace("Strategy", "").lower()


class JITFallbackMixin:
    """Mixin for JIT strategies that provides fallback functionality.
    
    This mixin adds methods for handling fallback cases when a JIT strategy
    cannot be applied or encounters errors during tracing/compilation.
    """

    def fallback_compile(self, func: F) -> F:
        """Provide a fallback compilation when the strategy cannot be applied.
        
        Args:
            func: The original function
            
        Returns:
            A function that executes the original function directly
        """
        return func
    
    def should_fallback(self, func: Callable) -> bool:
        """Determine whether compilation should fall back to the original function.
        
        Args:
            func: Function to analyze
            
        Returns:
            True if compilation should fall back, False otherwise
        """
        return False
