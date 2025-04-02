"""Adaptive execution system for Ember computational graphs.

Provides a high-performance, context-aware execution framework that optimizes
parallel operations within the XCS system. Core elements include executors specialized 
for different Ember workloads and a smart dispatcher that routes operations to the
most efficient execution mechanism based on their characteristics.
"""

import asyncio
import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, TypeVar, runtime_checkable
)

from ember.xcs.utils.execution_analyzer import ExecutionTracker

T = TypeVar('T')
U = TypeVar('U')
logger = logging.getLogger(__name__)


@runtime_checkable
class Executor(Protocol[T, U]):
    """Protocol for executors that run batched tasks."""
    
    def execute(self, fn: Callable[[T], U], inputs: List[T]) -> List[U]:
        """Execute function across inputs."""
        ...


class ThreadExecutor(Executor[Dict[str, Any], Any]):
    """Thread pool-based executor for Ember workloads with mixed I/O and computation.
    
    Optimized for model inference operations and data transformation tasks
    that involve both I/O (model API calls) and computation (data processing).
    Provides good performance for moderate levels of concurrency in the XCS graph.
    """
    
    def __init__(
        self, 
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self.fail_fast = fail_fast
        self._executor = None
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Create thread pool with lazy initialization."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
        
    def execute(self, fn: Callable, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across inputs using thread pool."""
        executor = self._get_executor()
        results = []
        
        # Submit all tasks
        futures = [
            executor.submit(lambda i=i: fn(inputs=i)) 
            for i in inputs
        ]
        
        # Collect results in submission order
        for _, future in enumerate(futures):
            try:
                result = future.result(timeout=self.timeout)
                results.append(result)
            except Exception as e:
                if not self.fail_fast:
                    logger.warning("Error in thread execution: %s", e)
                    results.append(None)
                else:
                    raise
                    
        return results
    
    def close(self) -> None:
        """Release executor resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


class AsyncExecutor(Executor[Dict[str, Any], Any]):
    """Asyncio-based executor for highly concurrent LLM API workloads.
    
    Specialized for Ember operations that make many concurrent model API requests,
    like parallel model inference or ensemble operations across multiple LLM providers.
    More efficient than threads when dealing with hundreds of concurrent operations
    or when precise concurrency control is needed.
    """
    
    def __init__(
        self, 
        max_concurrency: int = 20,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
    ):
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.fail_fast = fail_fast
        
    def execute(self, fn: Callable, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Execute function across inputs using asyncio."""
        return self._run_async_gather(fn, inputs)
    
    def _run_async_gather(
        self, fn: Callable, inputs: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute multiple calls concurrently with asyncio."""
        async def _gather_with_semaphore():
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrency)
            
            async def _process_item(inputs: Dict[str, Any]) -> Any:
                """Process single item with concurrency control."""
                async with semaphore:
                    if inspect.iscoroutinefunction(fn):
                        return await fn(inputs=inputs)
                    
                    # Run synchronous function in thread pool
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, lambda: fn(inputs=inputs)
                    )
            
            # Create tasks for all inputs
            tasks = [_process_item(i) for i in inputs]
            
            # Add timeout if specified (applied at the gather level)
            if self.timeout:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError as e:
                    logger.error("Execution timed out after %s seconds", self.timeout)
                    raise e
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
            # Process results, handling exceptions
            processed = []
            for result in results:
                if isinstance(result, Exception):
                    if not self.fail_fast:
                        logger.warning("Error in async execution: %s", result)
                        processed.append(None)
                    else:
                        raise result
                else:
                    processed.append(result)
                    
            return processed
        
        # Run the async gathering function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(_gather_with_semaphore())
        
    def close(self) -> None:
        """No resources to close in async executor."""
        pass


class Dispatcher:
    """Smart task dispatcher that optimizes XCS graph execution.
    
    Core component of Ember's execution system that analyzes functions to determine
    their execution characteristics and routes them to the optimal executor.
    Provides adaptive selection between thread pool (for general workloads) and
    asyncio (for high-concurrency API operations) to maximize throughput of the
    Ember computation graph.
    """
    
    def __init__(
        self, 
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
        executor: str = "auto",
    ):
        """Initialize dispatcher with execution parameters.
        
        Args:
            max_workers: Maximum worker threads or concurrent operations
            timeout: Optional timeout for operations in seconds
            fail_fast: If False, continues execution despite individual failures
            executor: Executor selection - "auto", "async", or "thread"
        """
        # Validate inputs
        if executor not in ("auto", "async", "thread"):
            raise ValueError(
                f"Invalid executor: {executor}. "
                f"Must be 'auto', 'async', or 'thread'."
            )
            
        self.max_workers = max_workers
        self.timeout = timeout
        self.fail_fast = fail_fast
        self.executor = executor
        
        # Create executors lazily
        self._thread_executor = None
        self._async_executor = None
        
    def _get_thread_executor(self) -> ThreadExecutor:
        """Get or create thread executor."""
        if self._thread_executor is None:
            self._thread_executor = ThreadExecutor(
                max_workers=self.max_workers,
                timeout=self.timeout,
                fail_fast=self.fail_fast,
            )
        return self._thread_executor
        
    def _get_async_executor(self) -> AsyncExecutor:
        """Get or create async executor."""
        if self._async_executor is None:
            self._async_executor = AsyncExecutor(
                max_concurrency=self.max_workers or 20,
                timeout=self.timeout,
                fail_fast=self.fail_fast,
            )
        return self._async_executor
    
    def _select_executor(self, fn: Callable) -> Executor:
        """Select optimal executor for Ember function.
        
        Analyzes each function to determine if it's an API/LLM operation (better suited 
        for AsyncExecutor) or a general Ember task (better with ThreadExecutor).
        Uses both static analysis of function names/modules and runtime performance
        metrics to make increasingly accurate decisions.
        
        Args:
            fn: Function to be executed
            
        Returns:
            Appropriate executor for the function
        """
        # Explicit executor selection takes precedence
        if self.executor == "async":
            return self._get_async_executor()
        if self.executor == "thread":
            return self._get_thread_executor()
            
        # Auto-detection based on function characteristics
        if inspect.iscoroutinefunction(fn):
            return self._get_async_executor()
            
        # Use adaptive smart detection 
        if ExecutionTracker.is_likely_io_bound(fn):
            return self._get_async_executor()
            
        # Default to thread executor for general Ember operations
        # (data processing, transformations, and mixed workloads)
        return self._get_thread_executor()
    
    def map(self, fn: Callable, inputs_list: List[Dict[str, Any]]) -> List[Any]:
        """Map function across Ember inputs with optimal parallelism.
        
        Core method for parallel execution in the XCS system. Automatically determines
        whether a function is best executed with threads (general Ember operations) or
        asyncio (LLM/API calls), routes to the proper executor, and collects results.
        
        Args:
            fn: Function to execute for each input (typically an Operator method)
            inputs_list: List of Ember input dictionaries
            
        Returns:
            Results from each parallel function execution
        """
        if not inputs_list:
            return []
        
        # Start timing the overall execution
        start_wall = time.time()
        start_cpu = time.process_time()
        
        try:
            # Select and use appropriate executor
            executor = self._select_executor(fn)
            return executor.execute(fn, inputs_list)
        finally:
            # Track execution time for future optimization
            end_cpu = time.process_time()
            end_wall = time.time()
            
            # Calculate execution metrics for optimization
            cpu_time = end_cpu - start_cpu
            wall_time = end_wall - start_wall
            
            # Record metrics to improve future execution strategy selection
            if inputs_list:
                ExecutionTracker.update_metrics(fn, cpu_time, wall_time)
    
    def close(self) -> None:
        """Release all allocated resources."""
        if self._thread_executor:
            self._thread_executor.close()
        if self._async_executor:
            self._async_executor.close()


# For compatibility with existing code that may have imported these names
TaskExecutor = Dispatcher
ExecutionCoordinator = Dispatcher
ThreadedStrategy = ThreadExecutor
AsyncStrategy = AsyncExecutor