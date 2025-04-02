"""
Tracing and Graph Building Infrastructure for XCS.

Provides tracing context management and data collection for building 
computational graphs from execution traces. This module is a core 
component of the XCS system that enables automatic graph construction.
"""

from ember.xcs.tracer._context_types import TraceContextData
from ember.xcs.tracer.xcs_tracing import TraceRecord, TracerContext
from ember.xcs.tracer.autograph import AutoGraphBuilder, autograph

__all__ = [
    # Core tracing system
    "TraceRecord",
    "TraceContextData",
    "TracerContext",
    
    # Graph building
    "AutoGraphBuilder",
    "autograph",
]