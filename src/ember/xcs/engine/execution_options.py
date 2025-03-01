"""
Execution Options Context for Ember.

This module provides a context manager for configuring execution options for operators,
such as scheduler type and parallelism parameters.
"""

from __future__ import annotations

import threading
from contextlib import ContextDecorator
from typing import Any, Dict, Optional, Type, Union

from ember.xcs.engine.xcs_engine import (
    IScheduler,
    TopologicalSchedulerWithParallelDispatch,
)


class ExecutionOptions(ContextDecorator):
    """Context manager for configuring operator execution options.

    This provides a way to set execution parameters (scheduler, worker count, etc.)
    that will be used when executing operators within the context.

    Attributes:
        scheduler (Union[str, IScheduler]): The scheduler to use for execution.
            Can be a string identifier ("parallel", "sequential") or an IScheduler instance.
        max_workers (Optional[int]): Maximum number of worker threads for parallel execution.
    """

    _local = threading.local()

    def __init__(
        self,
        *,
        scheduler: Union[str, IScheduler] = "parallel",
        max_workers: Optional[int] = None,
    ) -> None:
        """Initialize execution options.

        Args:
            scheduler: Scheduler to use for execution. Can be a string ("parallel", "sequential")
                or an IScheduler instance.
            max_workers: Maximum number of worker threads for parallel execution.
        """
        self.scheduler = scheduler
        self.max_workers = max_workers

    def __enter__(self) -> ExecutionOptions:
        """Enter the execution options context.

        Returns:
            The active execution options context.
        """
        self._set_current(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> Optional[bool]:
        """Exit the execution options context.

        Args:
            exc_type: Exception type, if any.
            exc_value: Exception value, if any.
            traceback: Traceback, if any.

        Returns:
            None.
        """
        self._clear_current()
        return None

    @classmethod
    def get_current(cls) -> Optional[ExecutionOptions]:
        """Get the current active execution options.

        Returns:
            The current execution options, or None if not in an execution options context.
        """
        return getattr(cls._local, "current", None)

    def get_scheduler(self) -> IScheduler:
        """Get a scheduler instance based on the current configuration.

        Returns:
            An IScheduler instance.
        """
        if isinstance(self.scheduler, IScheduler):
            return self.scheduler

        if self.scheduler == "sequential":
            # Import here to avoid circular imports
            from ember.xcs.engine.xcs_noop_scheduler import NoopScheduler

            return NoopScheduler()
        else:  # Default to parallel
            return TopologicalSchedulerWithParallelDispatch(
                max_workers=self.max_workers
            )

    def _set_current(self, ctx: ExecutionOptions) -> None:
        """Set the current execution options context.

        Args:
            ctx: The execution options context to set as current.
        """
        type(self)._local.current = ctx

    def _clear_current(self) -> None:
        """Clear the current execution options context."""
        type(self)._local.current = None


# Convenience function for use as context manager
def execution_options(
    *, scheduler: Union[str, IScheduler] = "parallel", max_workers: Optional[int] = None
) -> ExecutionOptions:
    """Create an execution options context.

    Args:
        scheduler: Scheduler to use for execution. Can be a string ("parallel", "sequential")
            or an IScheduler instance.
        max_workers: Maximum number of worker threads for parallel execution.

    Returns:
        An ExecutionOptions context manager.
    """
    return ExecutionOptions(scheduler=scheduler, max_workers=max_workers)
