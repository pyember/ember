"""Enhanced JIT compilation strategy.

Implements JIT compilation with improved parallelism detection and optimization.
This strategy combines aspects of both trace-based and structural analysis
approaches, making it particularly effective for complex operators with loop-based
patterns like ensembles.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

from ember.xcs.graph.graph_builder import EnhancedTraceGraphBuilder
from ember.xcs.jit.cache import JITCache
from ember.xcs.jit.strategies.base_strategy import BaseStrategy, Strategy, JITFallbackMixin

logger = logging.getLogger(__name__)


class EnhancedStrategy(BaseStrategy, JITFallbackMixin):
    """Enhanced JIT compilation strategy.
    
    Compiles operators using sophisticated analysis that identifies parallelization
    opportunities even within sequential execution patterns. This strategy is
    optimized for complex operators like ensembles.
    """
    
    def __init__(self) -> None:
        """Initialize the enhanced strategy."""
        self.graph_builder = EnhancedTraceGraphBuilder()
    
    
    def analyze(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """Analyze a function to determine if enhanced JIT is appropriate.
        
        Args:
            func: Function to analyze
            
        Returns:
            Dictionary with analysis results
        """
        import inspect
        
        features = self._extract_common_features(func)
        score = 0
        rationale = []
        
        # Check for ensemble patterns
        ensemble_indicators = ["ensemble", "aggregate", "combine", "accumulate"]
        if features["is_class"]:
            # Check for common ensemble operator naming patterns
            class_name = func.__name__.lower()
            if any(indicator in class_name for indicator in ensemble_indicators):
                score += 30
                rationale.append("Name suggests ensemble pattern")
            
            # Check for iteration methods/patterns
            if hasattr(func, "items") or hasattr(func, "__iter__"):
                score += 20
                rationale.append("Has iteration capability")
        
        # Check for code patterns indicating nested loops
        if features["has_source"] and features["source_lines"] > 5:
            try:
                source = inspect.getsource(func)
                loop_count = source.count("for ") + source.count("while ")
                if loop_count > 1:
                    score += 40
                    rationale.append(f"Contains multiple loops ({loop_count})")
                elif loop_count > 0:
                    score += 20
                    rationale.append("Contains loops")
            except Exception:
                pass
        
        # Enhanced JIT is a good general-purpose fallback
        score += 10
        rationale.append("Good general-purpose option")
        
        return {
            "score": score,
            "rationale": "; ".join(rationale),
            "features": features,
        }
    
    def compile(
        self, 
        func: Callable[..., Any], 
        sample_input: Optional[Dict[str, Any]] = None,
        force_trace: bool = False,
        recursive: bool = True,
        cache: Optional[JITCache] = None,
        **options: Any
    ) -> Callable[..., Any]:
        """Compile a function using enhanced JIT.
        
        Args:
            func: Function to compile
            sample_input: Optional sample input for eager compilation
            force_trace: Whether to force tracing on every call
            recursive: Whether to JIT compile nested calls
            cache: JIT cache to use
            **options: Additional options
            
        Returns:
            Compiled function
        """
        from ember.xcs.tracer.xcs_tracing import TracerContext
        
        # Use provided cache or get the default
        cache = self._get_cache(cache)
        
        @functools.wraps(func)
        def enhanced_function(*, inputs: Dict[str, Any]) -> Any:
            # Check if JIT is disabled
            if getattr(enhanced_function, "_jit_disabled", False) or force_trace:
                # Execute directly without JIT
                execution_start = time.time()
                result = func(inputs=inputs)
                execution_duration = time.time() - execution_start
                cache.metrics.record_execution(execution_duration)
                return result
            
            # Try to get state signature
            state_signature = None
            if hasattr(func, "get_state_signature") and callable(
                getattr(func, "get_state_signature")
            ):
                state_signature = func.get_state_signature()
            
            # Try to get compiled graph
            graph = cache.get_with_state(func, state_signature)
            
            if graph is not None:
                # Execute cached graph
                try:
                    from ember.xcs.jit.execution_utils import execute_compiled_graph
                    return execute_compiled_graph(graph, inputs, cache, func=func)
                except Exception as e:
                    logger.warning(f"Error executing graph: {e}. Falling back to direct execution.")
            
            # Trace execution if compilation fails or is disabled
            tracing_start = time.time()
            with TracerContext() as tracer:
                # Record recursive flag in tracer context
                tracer.enable_recursion = recursive
                
                # Execute the function and record trace
                execution_start = time.time()
                result = func(inputs=inputs)
                execution_duration = time.time() - execution_start
                cache.metrics.record_execution(execution_duration)
                
                # Build graph from trace if non-trivial
                if tracer.records and len(tracer.records) > 1:
                    compilation_start = time.time()
                    graph = self.graph_builder.build_graph(tracer.records)
                    compilation_duration = time.time() - compilation_start
                    cache.metrics.record_compilation(compilation_duration)
                    
                    # Cache compiled graph
                    state_signature = None
                    if hasattr(func, "get_state_signature") and callable(
                        getattr(func, "get_state_signature")
                    ):
                        state_signature = func.get_state_signature()
                    
                    cache.set_with_state(func, graph, state_signature)
            
            # Record tracing time
            tracing_duration = time.time() - tracing_start
            cache.metrics.record_tracing(tracing_duration)
            
            return result
        
        # Add control methods
        self._add_control_methods(enhanced_function, func, cache)
        
        return enhanced_function