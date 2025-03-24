# Export the autograph context manager for public API
from contextlib import contextmanager

from ._context_types import TraceContextData
from .autograph import AutoGraphBuilder
from .xcs_tracing import TracerContext, TraceRecord


@contextmanager
def autograph(*args, **kwargs):
    """Context manager for automatic graph building.

    Creates a computation graph context where operations are automatically
    recorded as graph nodes rather than being executed immediately.

    Yields:
        An XCSGraph object that can be used to execute the recorded operations
    """
    from ember.xcs.graph.xcs_graph import XCSGraph

    graph = XCSGraph()
    try:
        yield graph
    finally:
        # When exiting context, the graph has been built and can be used
        pass


__all__ = [
    "TracerContext",
    "TraceRecord",
    "TraceContextData",
    "AutoGraphBuilder",
    "autograph",
]
