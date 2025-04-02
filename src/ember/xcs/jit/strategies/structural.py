"""Structure-based JIT compilation strategy.

Implements JIT compilation by analyzing the structure of operators directly
without executing them. This strategy is particularly effective for container
operators with nested sub-operators, as it can detect parallelization opportunities
based on the operator structure.
"""

import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, Optional

from ember.xcs.graph.graph_builder import StructuralGraphBuilder
from ember.xcs.jit.cache import JITCache
from ember.xcs.jit.strategies.base_strategy import BaseStrategy, Strategy, JITFallbackMixin

logger = logging.getLogger(__name__)


class StructuralStrategy(BaseStrategy, JITFallbackMixin):
    """Structure-based JIT compilation strategy.
    
    Compiles operators by analyzing their structure directly, without
    execution tracing. This approach is particularly effective for
    container operators with nested sub-operators.
    """
    
    def __init__(self) -> None:
        """Initialize the structural strategy."""
        self.graph_builder = StructuralGraphBuilder()
    
    
    def analyze(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """Analyze a function to determine if structural JIT is appropriate.
        
        Args:
            func: Function to analyze
            
        Returns:
            Dictionary with analysis results
        """
        features = self._extract_common_features(func)
        score = 0
        rationale = []
        
        # Check if it's an operator class with forward method
        if features["is_class"]:
            # Check for operator characteristics
            if hasattr(func, "forward") and callable(getattr(func, "forward", None)):
                score += 30
                rationale.append("Has 'forward' method (likely an operator)")
            
            # Check for nested operators
            has_operator_fields = False
            for attr_name in dir(func):
                if attr_name.startswith("_"):
                    continue
                
                attr = getattr(func, attr_name, None)
                if inspect.isclass(attr) and hasattr(attr, "forward"):
                    has_operator_fields = True
                    break
            
            if has_operator_fields:
                score += 40
                rationale.append("Has nested operator fields (container pattern)")
        
        # Check for specific method signatures that indicate operator composition
        if hasattr(func, "__call__") and callable(getattr(func, "__call__")):
            score += 20
            rationale.append("Has __call__ method")
        
        # Check for known operator-specific attributes
        if hasattr(func, "specification"):
            score += 10
            rationale.append("Has 'specification' attribute (operator pattern)")
        
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
        """Compile a function using structural JIT.
        
        Args:
            func: Function to compile
            sample_input: Optional sample input (not used in structural JIT)
            force_trace: Whether to force analysis on every call
            recursive: Whether to recursively analyze nested operators
            cache: JIT cache to use
            **options: Additional options
            
        Returns:
            Compiled function
        """
        # Use provided cache or get the default
        cache = self._get_cache(cache)
        
        @functools.wraps(func)
        def structural_function(*, inputs: Dict[str, Any]) -> Any:
            # Check if JIT is disabled
            if getattr(structural_function, "_jit_disabled", False) or force_trace:
                # Execute directly
                execution_start = time.time()
                result = func(inputs=inputs)
                execution_duration = time.time() - execution_start
                cache.metrics.record_execution(execution_duration)
                return result
            
            # Try to get structure signature
            state_signature = None
            if hasattr(func, "get_structure_signature") and callable(
                getattr(func, "get_structure_signature")
            ):
                state_signature = func.get_structure_signature()
            
            # Try to get cached graph
            graph = cache.get_with_state(func, state_signature)
            
            if graph is not None:
                # Execute cached graph
                try:
                    from ember.xcs.jit.execution_utils import execute_compiled_graph
                    return execute_compiled_graph(graph, inputs, cache, func=func)
                except Exception as e:
                    logger.warning(f"Error executing graph: {e}. Falling back to direct execution.")
            
            # Analyze structure and build graph
            compilation_start = time.time()
            
            # Set recursion flag if available on the graph builder
            if hasattr(self.graph_builder, "set_recursive"):
                self.graph_builder.set_recursive(recursive)
                
            graph = self.graph_builder.build_graph(func)
            compilation_duration = time.time() - compilation_start
            cache.metrics.record_compilation(compilation_duration)
            
            # Cache compiled graph
            state_signature = None
            if hasattr(func, "get_structure_signature") and callable(
                getattr(func, "get_structure_signature")
            ):
                state_signature = func.get_structure_signature()
            
            cache.set_with_state(func, graph, state_signature)
            
            # Execute and return with fallback
            return self.execute_with_fallback(graph, func, inputs, cache)
        
        # Add control methods
        self._add_control_methods(structural_function, func, cache)
        
        return structural_function
    
    
    
    def execute_with_fallback(
        self, 
        graph: Any, 
        original_func: Callable, 
        inputs: Dict[str, Any], 
        cache: JITCache
    ) -> Any:
        """Execute a compiled graph with fallback to direct execution.
        
        Args:
            graph: Compiled graph
            original_func: Original function
            inputs: Input values
            cache: JIT cache
            
        Returns:
            Execution result
        """
        try:
            # Try to execute the compiled graph
            from ember.xcs.jit.execution_utils import execute_compiled_graph
            return execute_compiled_graph(graph, inputs, cache, func=original_func)
        except Exception as e:
            # Log error and fall back to direct execution
            logger.warning(f"Error executing compiled graph: {e}. Falling back to direct execution.")
            
            # Execute directly and time it
            execution_start = time.time()
            result = original_func(inputs=inputs)
            execution_duration = time.time() - execution_start
            cache.metrics.record_execution(execution_duration)
            
            return result