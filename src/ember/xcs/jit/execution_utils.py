"""Execution utilities for JIT-compiled graphs.

Provides utilities for executing compiled graphs with various strategies.
This module bridges between the JIT compilation system and the engine's execution
capabilities, handling input/output conversion, error handling, and metrics.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

from ember.xcs.common.plans import ExecutionResult
from ember.xcs.engine.unified_engine import execute_graph, ExecutionOptions
from ember.xcs.jit.cache import JITCache
from ember.xcs.schedulers.factory import create_scheduler

logger = logging.getLogger(__name__)


def execute_compiled_graph(
    graph: Any, 
    inputs: Dict[str, Any], 
    cache: JITCache, 
    func: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a compiled graph with provided inputs.
    
    Args:
        graph: Compiled graph to execute
        inputs: Input values to the graph
        cache: JIT cache for metrics tracking
        options: Optional execution options
        
    Returns:
        Dictionary with execution results
    """
    # Track execution time for metrics
    execution_start = time.time()
    
    try:
        # If graph has a specific execution mode attached, use it
        if hasattr(graph, "execution_mode") and hasattr(graph, "execution_options"):
            mode = getattr(graph, "execution_mode", "auto")
            mode_options = getattr(graph, "execution_options", {})
            
            # Create execution options
            exec_options = ExecutionOptions(
                scheduler_type=mode,
                **mode_options
            )
        else:
            # Use default options or provided options
            exec_options = ExecutionOptions(**(options or {}))
        
        # Execute the graph
        result_dict = execute_graph(graph, inputs, options=exec_options)
        
        # Determine root node or output node
        root_id = None
        if hasattr(graph, "root_id"):
            root_id = graph.root_id
        elif hasattr(graph, "_output_node_id"):
            root_id = graph._output_node_id
        elif hasattr(graph, "metadata") and "root_id" in graph.metadata:
            root_id = graph.metadata["root_id"]
        elif hasattr(graph, "metadata") and "output_node_id" in graph.metadata:
            root_id = graph.metadata["output_node_id"]
            
        # Get result from appropriate node
        if root_id and root_id in result_dict:
            result = result_dict.get(root_id, {})
        else:
            # If no specific output node, return all results
            result = result_dict
            # If the result is empty, call the original function directly
            if not result and func is not None:
                return func(inputs=inputs)
            
        # Ensure proper boundary crossing for outputs
        # If the original function has a specification, validate output
        if func is not None and hasattr(func, "specification") and hasattr(func.specification, "validate_output"):
            try:
                result = func.specification.validate_output(output=result)
            except Exception as e:
                import logging
                logging.warning(f"Output validation failed in execute_compiled_graph: {e}")
        
        # Record execution time
        execution_duration = time.time() - execution_start
        if func is not None:
            func_id = id(func)
            cache.metrics.record_execution(execution_duration, func_id)
        else:
            cache.metrics.record_execution(execution_duration)
        
        return result
    except Exception as e:
        # Log error and record execution time (failure case)
        execution_duration = time.time() - execution_start
        if func is not None:
            func_id = id(func)
            cache.metrics.record_execution(execution_duration, func_id)
        else:
            cache.metrics.record_execution(execution_duration)
        
        # Propagate exception with context
        logger.error(f"Error executing JIT graph: {e}")
        raise