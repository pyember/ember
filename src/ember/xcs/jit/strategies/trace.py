"""Trace-based JIT compilation strategy.

Implements JIT compilation by tracing function execution to capture the
computation graph dynamically. This approach is most effective for operators
with data-dependent execution patterns.
"""

import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, Optional

from ember.xcs.jit.cache import JITCache
from ember.xcs.jit.strategies.base_strategy import BaseStrategy, Strategy, JITFallbackMixin

logger = logging.getLogger(__name__)


class TraceStrategy(BaseStrategy, JITFallbackMixin):
    """Trace-based JIT compilation strategy.
    
    Compiles operators by executing them with tracing enabled and recording
    the execution graph. This approach works well for operators with simple
    execution patterns.
    """
    
    def analyze(self, func: Callable) -> Dict[str, Any]:
        """Analyze a function to determine if trace-based JIT is appropriate.
        
        Args:
            func: Function to analyze
            
        Returns:
            Dictionary with analysis results
        """
        import inspect
        
        # Extract basic features
        features = self._extract_common_features(func)
        score = 0
        rationale = []
        
        # Trace-based JIT is good for simple functions
        if features["source_lines"] < 20:
            score += 30
            rationale.append("Simple function (< 20 lines)")
            
        # Check for simple conditional patterns
        if features["has_source"]:
            # The source is not directly stored in features, but we can use inspect again
            try:
                source = inspect.getsource(func)
                if source.count("if ") < 3 and source.count("for ") < 2:
                    score += 20
                    rationale.append("Simple control flow")
            except Exception:
                pass
                
        # Trace is the most basic strategy, so it has a base score
        score += 5
        rationale.append("Basic fallback strategy")
            
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
        """Compile a function using trace-based JIT.
        
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
        from ember.xcs.graph.xcs_graph import XCSGraph
        
        # Get or create a cache
        cache = self._get_cache(cache)
        
        @functools.wraps(func)
        def traced_function(*, inputs: Dict[str, Any]) -> Any:
            # Check if JIT is disabled or if we're forcing a trace
            if getattr(traced_function, "_jit_disabled", False) or force_trace:
                # Execute directly without JIT
                execution_start = time.time()
                result = func(inputs=inputs)
                execution_duration = time.time() - execution_start
                cache.metrics.record_execution(execution_duration)
                return result
                
            # Try to get cached graph
            graph = cache.get(func)
            
            if graph is not None:
                # Execute cached graph
                execution_start = time.time()
                try:
                    from ember.xcs.engine import execute_graph
                    results = execute_graph(graph, inputs)
                    execution_end = time.time()
                    execution_duration = execution_end - execution_start
                    cache.metrics.record_execution(execution_duration)
                    
                    # Find the output node (last node in the graph)
                    output_node_id = graph.get_output_nodes()[0]
                    return results[output_node_id]
                except Exception as e:
                    logger.warning(f"Error executing cached graph: {e}. Falling back to direct execution.")
            
            # Trace execution
            tracing_start = time.time()
            graph = XCSGraph()
            
            with TracerContext() as tracer:
                # Enable recursive tracing if requested
                tracer.enable_recursion = recursive
                
                # Execute and record trace
                execution_start = time.time()
                result = func(inputs=inputs)
                execution_end = time.time()
                execution_duration = execution_end - execution_start
                cache.metrics.record_execution(execution_duration)
                
                # Build graph from trace if trace was recorded
                if tracer.records:
                    compilation_start = time.time()
                    
                    # Add all nodes first
                    for i, record in enumerate(tracer.records):
                        # Create a deterministic function that replays the recorded output
                        def create_replay_function(rec):
                            def replay_func(**_):
                                # Simply replay the recorded outputs for this operation
                                return rec.outputs
                            return replay_func
                        
                        # Add node with proper metadata for diagnostics and optimization
                        node_metadata = {
                            "original_operator": record.operator_name,
                            "instance_id": record.instance_id,
                            "execution_time": record.duration,
                            "timestamp": record.timestamp
                        }
                        
                        # Use the original node_id for traceability
                        node_id = record.node_id or f"traced_node_{i}"
                        
                        # Add the node to the graph
                        graph.add_node(
                            operator=create_replay_function(record),
                            node_id=node_id,
                            name=record.operator_name,
                            metadata=node_metadata
                        )
                    
                    # Add dependencies between nodes
                    for i in range(1, len(tracer.records)):
                        prev_record = tracer.records[i-1]
                        curr_record = tracer.records[i]
                        graph.add_edge(prev_record.node_id, curr_record.node_id)
                        
                    compilation_end = time.time()
                    compilation_duration = compilation_end - compilation_start
                    cache.metrics.record_compilation(compilation_duration)
                    
                    # Cache the graph
                    cache.set(func, graph)
            
            # Record tracing time
            tracing_end = time.time()
            tracing_duration = tracing_end - tracing_start
            cache.metrics.record_tracing(tracing_duration)
            
            return result
        
        # Add control methods
        self._add_control_methods(traced_function, func, cache)
        
        return traced_function
    
