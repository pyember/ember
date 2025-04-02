"""Unified execution engine for XCS graphs.

Provides a comprehensive execution system for computational graphs with
support for various scheduling strategies and execution modes.
"""

import dataclasses
import logging
import time
from typing import Any, Callable, Dict, Optional, Union

from ember.xcs.common.plans import ExecutionResult
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.schedulers.factory import create_scheduler
from ember.xcs.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


# Import the canonical ExecutionOptions from execution_options.py
# to maintain a single source of truth
from ember.xcs.engine.execution_options import ExecutionOptions as BaseExecutionOptions

# Define a class that adapts BaseExecutionOptions for use in unified_engine.py
class ExecutionOptions:
    """Options controlling graph execution behavior.
    
    This adapter class converts between the unified engine's execution options
    and the core engine's execution options, ensuring compatibility between systems
    while maintaining a single configuration point.
    """
    
    def __init__(
        self,
        scheduler: str = "auto",
        max_workers: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        use_parallel: bool = True,
        continue_on_error: bool = False,
        return_partial_results: bool = True,
        **additional_options: Any
    ):
        """Initialize execution options, compatible with unified engine.
        
        Args:
            scheduler: Type of scheduler to use
            max_workers: Maximum number of worker threads
            timeout_seconds: Maximum execution time before timeout
            use_parallel: Whether to enable parallelism at all
            continue_on_error: Whether to continue after node errors
            return_partial_results: Whether to return partial results on timeout/error
            **additional_options: Additional options for specialized schedulers
        """
        # Handle scheduler_type for backward compatibility
        if "scheduler_type" in additional_options:
            scheduler = additional_options.pop("scheduler_type")
            
        # Create a base options instance with compatible parameters
        self._base_options = BaseExecutionOptions(
            scheduler=scheduler,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
            use_parallel=use_parallel,
            **additional_options
        )
        
        # Store engine-specific options
        self.continue_on_error = continue_on_error
        self.return_partial_results = return_partial_results
        
    # Forward compatibility properties
    @property
    def scheduler(self) -> str:
        return self._base_options.scheduler
        
    @property
    def scheduler_type(self) -> str:
        # For backward compatibility
        return self._base_options.scheduler
        
    @property
    def max_workers(self) -> Optional[int]:
        return self._base_options.max_workers
        
    @property
    def timeout_seconds(self) -> Optional[float]:
        return self._base_options.timeout_seconds
        
    @property
    def use_parallel(self) -> bool:
        return self._base_options.use_parallel
        
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying options."""
        if hasattr(self._base_options, name):
            return getattr(self._base_options, name)
        raise AttributeError(f"'ExecutionOptions' has no attribute '{name}'")
        
    def to_base_options(self) -> BaseExecutionOptions:
        """Convert to the base options type for interoperability."""
        return self._base_options


class GraphExecutor:
    """Core execution engine for XCS graphs.
    
    Manages the execution of computational graphs using various scheduling
    strategies, handling errors, timeouts, and partial results.    
    """
    
    def __init__(self) -> None:
        """Initialize the graph executor."""
        pass
    
    def execute(self,
                graph: XCSGraph,
                inputs: Dict[str, Any],
                options: Optional[ExecutionOptions] = None,
                scheduler: Optional[BaseScheduler] = None) -> Dict[str, Dict[str, Any]]:
        """Execute a computational graph with provided inputs.
        
        Args:
            graph: Graph to execute
            inputs: Input values for graph execution
            options: Execution options
            scheduler: Optional explicit scheduler to use
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        # Use default options if none provided
        if options is None:
            options = ExecutionOptions()
        
        # Create scheduler if not explicitly provided
        if scheduler is None:
            # Check if parallelism is disabled
            if not options.use_parallel and options.scheduler == "auto":
                # Force sequential scheduler for disabled parallelism
                scheduler_type = "sequential"
            else:
                scheduler_type = options.scheduler
                
            # Create appropriate scheduler
            scheduler = create_scheduler(
                scheduler_type,
                max_workers=options.max_workers
            )
        
        # Execute graph with timeout handling
        start_time = time.perf_counter()
        try:
            # Prepare for execution
            scheduler.prepare(graph)
            
            # Execute with scheduler
            results = scheduler.execute(graph, inputs)
            
            return results
        except Exception as e:
            logger.error(f"Error executing graph: {e}")
            if options.return_partial_results:
                # Return any results obtained before error
                return scheduler.get_partial_results()
            raise
        finally:
            duration = time.perf_counter() - start_time
            logger.debug(f"Graph execution complete in {duration:.4f} seconds")


# Singleton executor instance
_executor = GraphExecutor()


def execute_graph(
    graph: XCSGraph,
    inputs: Dict[str, Any],
    options: Optional[Union[ExecutionOptions, BaseExecutionOptions, Dict[str, Any]]] = None,
    scheduler: Optional[BaseScheduler] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute a computational graph with provided inputs.
    
    This is the main entry point for graph execution in the XCS system.    
    
    Args:
        graph: Graph to execute
        inputs: Input values for graph execution
        options: Execution options - can be an ExecutionOptions object, BaseExecutionOptions, 
                 or a dict of options
        scheduler: Optional explicit scheduler to use
        
    Returns:
        Dictionary mapping node IDs to their output results
    """
    # Handle different option types for flexibility
    if options is None:
        # Default options
        effective_options = ExecutionOptions()
    elif isinstance(options, dict):
        # Convert dictionary to options
        effective_options = ExecutionOptions(**options)
    elif isinstance(options, BaseExecutionOptions):
        # Convert base options to engine options
        effective_options = ExecutionOptions(
            scheduler=options.scheduler,
            max_workers=options.max_workers, 
            timeout_seconds=options.timeout_seconds,
            use_parallel=options.use_parallel
        )
    else:
        # Already the right type
        effective_options = options
    
    # Use the singleton executor for consistency
    return _executor.execute(graph, inputs, effective_options, scheduler)


class ExecutionMetrics:
    """Metrics for graph execution performance."""
    
    def __init__(self) -> None:
        """Initialize execution metrics."""
        self.execution_time_ms: float = 0.0
        self.node_count: int = 0
        self.scheduler_overhead_ms: float = 0.0


class execution_options:
    """Context manager for setting default execution options.
    
    Example:
    ```python
    with execution_options(scheduler="wave", max_workers=4):
        # All graph executions within this context use these options
        result1 = execute_graph(graph1, inputs1)
        result2 = execute_graph(graph2, inputs2)
    ```
    """
    
    # Thread-local storage for context stack
    _context_stack = []
    
    def __init__(self, **kwargs) -> None:
        """Initialize with execution options.
        
        Args:
            **kwargs: Execution option settings
        """
        self.options = kwargs
    
    def __enter__(self) -> None:
        """Enter context, pushing options onto stack."""
        execution_options._context_stack.append(self.options)
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context, popping options from stack."""
        execution_options._context_stack.pop()
    
    @staticmethod
    def get_current_options() -> Dict[str, Any]:
        """Get current options from context stack.
        
        Returns:
            Dictionary of current execution options
        """
        if not execution_options._context_stack:
            return {}
        return execution_options._context_stack[-1]