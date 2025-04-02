"""JIT compilation caching system.

Manages compiled graphs and execution metrics for Just-In-Time compilation,
supporting invalidation, metrics collection, and state signature handling.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Global JIT cache instance for default use
_GLOBAL_CACHE = None
_CACHE_LOCK = threading.Lock()


class JITMetrics:
    """Metrics for JIT compilation and execution performance.
    
    Tracks compilation times, execution times, cache hits/misses, and other
    metrics to help with performance analysis and optimization.
    """
    
    def __init__(self) -> None:
        """Initialize empty metrics."""
        # Time metrics in seconds
        self.total_tracing_time = 0.0
        self.total_compilation_time = 0.0
        self.total_execution_time = 0.0
        
        # Operation counts
        self.cache_hits = 0
        self.cache_misses = 0
        self.compilation_count = 0
        self.execution_count = 0
        
        # Function-specific metrics
        self.function_metrics: Dict[int, Dict[str, Any]] = {}
    
    def record_tracing(self, duration: float, function_id: Optional[int] = None) -> None:
        """Record a tracing operation's duration.
        
        Args:
            duration: Duration in seconds
            function_id: Optional ID for function-specific metrics
        """
        self.total_tracing_time += duration
        
        if function_id is not None:
            if function_id not in self.function_metrics:
                self.function_metrics[function_id] = {
                    "tracing_time": 0.0,
                    "compilation_time": 0.0,
                    "execution_time": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "compilation_count": 0,
                    "execution_count": 0,
                }
            self.function_metrics[function_id]["tracing_time"] += duration
    
    def record_compilation(self, duration: float, function_id: Optional[int] = None) -> None:
        """Record a compilation operation's duration.
        
        Args:
            duration: Duration in seconds
            function_id: Optional ID for function-specific metrics
        """
        self.total_compilation_time += duration
        self.compilation_count += 1
        
        if function_id is not None:
            if function_id not in self.function_metrics:
                self.function_metrics[function_id] = {
                    "tracing_time": 0.0,
                    "compilation_time": 0.0,
                    "execution_time": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "compilation_count": 0,
                    "execution_count": 0,
                }
            self.function_metrics[function_id]["compilation_time"] += duration
            self.function_metrics[function_id]["compilation_count"] += 1
    
    def record_execution(self, duration: float, function_id: Optional[int] = None) -> None:
        """Record an execution operation's duration.
        
        Args:
            duration: Duration in seconds
            function_id: Optional ID for function-specific metrics
        """
        self.total_execution_time += duration
        self.execution_count += 1
        
        if function_id is not None:
            if function_id not in self.function_metrics:
                self.function_metrics[function_id] = {
                    "tracing_time": 0.0,
                    "compilation_time": 0.0,
                    "execution_time": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "compilation_count": 0,
                    "execution_count": 0,
                }
            self.function_metrics[function_id]["execution_time"] += duration
            self.function_metrics[function_id]["execution_count"] += 1
            
    def record_cache_hit(self, function_id: Optional[int] = None) -> None:
        """Record a cache hit.
        
        Args:
            function_id: Optional ID for function-specific metrics
        """
        self.cache_hits += 1
        if function_id is not None:
            if function_id not in self.function_metrics:
                self.function_metrics[function_id] = {
                    "tracing_time": 0.0,
                    "compilation_time": 0.0,
                    "execution_time": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "compilation_count": 0,
                    "execution_count": 0,
                }
            self.function_metrics[function_id]["cache_hits"] += 1
            
    def record_cache_miss(self, function_id: Optional[int] = None) -> None:
        """Record a cache miss.
        
        Args:
            function_id: Optional ID for function-specific metrics
        """
        self.cache_misses += 1
        if function_id is not None:
            if function_id not in self.function_metrics:
                self.function_metrics[function_id] = {
                    "tracing_time": 0.0,
                    "compilation_time": 0.0,
                    "execution_time": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "compilation_count": 0,
                    "execution_count": 0,
                }
            self.function_metrics[function_id]["cache_misses"] += 1
    
    def get_function_metrics(self, function_id: int) -> Dict[str, Any]:
        """Get metrics for a specific function.
        
        Args:
            function_id: ID of the function
            
        Returns:
            Dictionary of metrics or empty dict if not tracked
        """
        return self.function_metrics.get(function_id, {})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        # Calculate derived metrics
        hit_rate = 0.0
        if (self.cache_hits + self.cache_misses) > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            
        avg_compilation = 0.0
        if self.compilation_count > 0:
            avg_compilation = self.total_compilation_time / self.compilation_count
            
        avg_execution = 0.0
        if self.execution_count > 0:
            avg_execution = self.total_execution_time / self.execution_count
            
        return {
            "total_tracing_time_ms": self.total_tracing_time * 1000,
            "total_compilation_time_ms": self.total_compilation_time * 1000,
            "total_execution_time_ms": self.total_execution_time * 1000,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "compilation_count": self.compilation_count,
            "execution_count": self.execution_count,
            "avg_compilation_time_ms": avg_compilation * 1000,
            "avg_execution_time_ms": avg_execution * 1000,
            "function_count": len(self.function_metrics),
        }


class JITCache:
    """Cache for JIT-compiled execution graphs.
    
    Provides caching for compiled graphs with support for invalidation,
    operator state tracking, and metrics collection.
    """
    
    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.metrics = JITMetrics()
        
    def get(self, key: Callable) -> Optional[Any]:
        """Get a cached graph for a function.
        
        Args:
            key: Function to look up
            
        Returns:
            Cached graph or None if not found
        """
        key_id = id(key)
        with self._lock:
            if key_id in self._cache and "graph" in self._cache[key_id]:
                self.metrics.record_cache_hit(key_id)
                return self._cache[key_id]["graph"]
                
        self.metrics.record_cache_miss(key_id)
        return None
    
    def set(self, key: Callable, value: Any) -> None:
        """Cache a graph for a function.
        
        Args:
            key: Function as cache key
            value: Graph to cache
        """
        key_id = id(key)
        with self._lock:
            if key_id not in self._cache:
                self._cache[key_id] = {}
            self._cache[key_id]["graph"] = value
            self._cache[key_id]["timestamp"] = time.time()
    
    def get_with_state(self, key: Callable, state_signature: Optional[str] = None) -> Optional[Any]:
        """Get a cached graph that matches both function and state.
        
        Args:
            key: Function to look up
            state_signature: Optional state signature for state-dependent caching
            
        Returns:
            Cached graph or None if not found/matched
        """
        key_id = id(key)
        with self._lock:
            if key_id in self._cache and "graphs_by_state" in self._cache[key_id]:
                if state_signature in self._cache[key_id]["graphs_by_state"]:
                    self.metrics.record_cache_hit(key_id)
                    return self._cache[key_id]["graphs_by_state"][state_signature]
                
        self.metrics.record_cache_miss(key_id)
        return None
    
    def set_with_state(self, key: Callable, value: Any, state_signature: Optional[str] = None) -> None:
        """Cache a graph for a function with associated state signature.
        
        Args:
            key: Function as cache key
            value: Graph to cache
            state_signature: Optional state signature for state-dependent caching
        """
        key_id = id(key)
        if state_signature is None:
            state_signature = "default"
            
        with self._lock:
            if key_id not in self._cache:
                self._cache[key_id] = {"graphs_by_state": {}}
            if "graphs_by_state" not in self._cache[key_id]:
                self._cache[key_id]["graphs_by_state"] = {}
                
            self._cache[key_id]["graphs_by_state"][state_signature] = value
            self._cache[key_id]["timestamp"] = time.time()
    
    def invalidate(self, key: Callable) -> None:
        """Invalidate all cached graphs for a function.
        
        Args:
            key: Function to invalidate
        """
        key_id = id(key)
        with self._lock:
            if key_id in self._cache:
                del self._cache[key_id]
                
    def invalidate_all(self) -> None:
        """Invalidate the entire cache."""
        with self._lock:
            self._cache.clear()
            
    def get_metrics(self, func: Optional[Callable] = None) -> Dict[str, Any]:
        """Get metrics for a function or overall cache metrics.
        
        Args:
            func: Optional function to get metrics for
            
        Returns:
            Dictionary with metric information
        """
        if func is not None:
            func_id = id(func)
            return self.metrics.get_function_metrics(func_id)
        return self.metrics.get_summary()
    
    def __len__(self) -> int:
        """Get number of cached functions.
        
        Returns:
            Number of cached functions
        """
        with self._lock:
            return len(self._cache)


def get_cache() -> JITCache:
    """Get the global JIT cache instance.
    
    Returns:
        Global cache instance
    """
    global _GLOBAL_CACHE
    with _CACHE_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = JITCache()
        return _GLOBAL_CACHE


def set_cache(cache: JITCache) -> None:
    """Set the global JIT cache instance.
    
    Args:
        cache: Cache instance to use globally
    """
    global _GLOBAL_CACHE
    with _CACHE_LOCK:
        _GLOBAL_CACHE = cache