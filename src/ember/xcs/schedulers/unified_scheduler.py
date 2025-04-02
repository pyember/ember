"""Unified scheduler implementations for XCS graph execution.

Provides concrete scheduler implementations using the strategy pattern,
with predefined combinations of ordering and execution strategies.
"""

from typing import Any, Dict, List, Optional

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.schedulers.base_scheduler import BaseScheduler
from ember.xcs.schedulers.base_scheduler_impl import (
    BaseSchedulerImpl,
    TopologicalOrderingStrategy,
    DepthOrderingStrategy,
    WaveOrderingStrategy,
    SequentialExecutionStrategy,
    ParallelExecutionStrategy,
    NoopExecutionStrategy
)


class NoOpScheduler(BaseSchedulerImpl):
    """Scheduler that doesn't actually execute operations.
    
    Useful for debugging, testing, and checking graph structure.
    """
    
    def __init__(self) -> None:
        """Initialize no-op scheduler."""
        super().__init__(
            ordering_strategy=TopologicalOrderingStrategy(),
            execution_strategy=NoopExecutionStrategy()
        )


class SequentialScheduler(BaseSchedulerImpl):
    """Scheduler that executes operations sequentially in topological order.
    
    Simple, safe execution for any graph, with no parallelism.
    """
    
    def __init__(self) -> None:
        """Initialize sequential scheduler."""
        super().__init__(
            ordering_strategy=TopologicalOrderingStrategy(),
            execution_strategy=SequentialExecutionStrategy()
        )


class TopologicalScheduler(BaseSchedulerImpl):
    """Scheduler that executes operations in topological order.
    
    Ensures correct dependency ordering while offering potential for parallelism.
    """
    
    def __init__(self) -> None:
        """Initialize topological scheduler."""
        super().__init__(
            ordering_strategy=TopologicalOrderingStrategy(),
            execution_strategy=SequentialExecutionStrategy()
        )


class ParallelScheduler(BaseSchedulerImpl):
    """Scheduler that executes operations in parallel where possible.
    
    Uses wave-based execution for maximum parallelism based on dependencies.
    """
    
    def __init__(self, max_workers: Optional[int] = None) -> None:
        """Initialize parallel scheduler.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        super().__init__(
            ordering_strategy=WaveOrderingStrategy(),
            execution_strategy=ParallelExecutionStrategy(max_workers=max_workers)
        )
        self.max_workers = max_workers


class WaveScheduler(BaseSchedulerImpl):
    """Scheduler that groups operations into execution waves.
    
    Provides structured parallelism with wave-based synchronization.
    """
    
    def __init__(self, max_workers: Optional[int] = None) -> None:
        """Initialize wave scheduler.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        super().__init__(
            ordering_strategy=WaveOrderingStrategy(),
            execution_strategy=ParallelExecutionStrategy(max_workers=max_workers)
        )
        self.max_workers = max_workers


# Legacy aliases
TraceScheduler = SequentialScheduler
TopologicalSchedulerWithParallelDispatch = ParallelScheduler