"""
Scheduler factory for XCS.

Provides a factory function for creating scheduler instances based on
scheduler type and options. This centralizes scheduler creation logic
and allows for dynamic selection of the most appropriate scheduler.
"""

import logging
from typing import Any, Dict, Optional

from ember.xcs.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


def create_scheduler(scheduler_type: str = "auto", **kwargs: Any) -> BaseScheduler:
    """Create a scheduler instance based on the specified type.
    
    Factory function that returns an appropriate scheduler for the given type
    and parameters.
    
    Args:
        scheduler_type: Type of scheduler to create. One of:
            - "auto": Automatically select the best scheduler
            - "sequential": Simple sequential execution
            - "parallel": Parallel execution with wave scheduling
            - "topological": Topological sort-based execution
            - "wave": Wave-based parallel execution
            - "noop": No-op scheduler for testing
        **kwargs: Additional scheduler-specific options
            - max_workers: Maximum number of parallel workers
            - timeout_seconds: Execution timeout
            - continue_on_error: Whether to continue after errors
            
    Returns:
        Scheduler instance
    """
    # Import implementations here to avoid circular imports
    from ember.xcs.schedulers.unified_scheduler import (
        NoOpScheduler,
        ParallelScheduler,
        SequentialScheduler,
        TopologicalScheduler,
        WaveScheduler,
    )
    
    # Extract common options
    max_workers = kwargs.get("max_workers")
    
    # Create appropriate scheduler based on type
    if scheduler_type == "noop":
        return NoOpScheduler()
    elif scheduler_type == "sequential":
        return SequentialScheduler()
    elif scheduler_type == "topological":
        return TopologicalScheduler()
    elif scheduler_type == "wave":
        return WaveScheduler(max_workers=max_workers)
    elif scheduler_type == "parallel":
        return ParallelScheduler(max_workers=max_workers)
    elif scheduler_type == "auto":
        # Auto-select based on properties
        if max_workers is not None and max_workers > 1:
            logger.debug("Automatically selecting wave scheduler for parallel execution")
            return WaveScheduler(max_workers=max_workers)
        else:
            logger.debug("Automatically selecting topological scheduler for sequential execution")
            return TopologicalScheduler()
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, falling back to topological")
        return TopologicalScheduler()