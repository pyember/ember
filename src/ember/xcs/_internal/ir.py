"""Backward-compatible IR aliases for legacy imports."""

from ember.xcs.compiler.graph import (
    GraphParallelismAnalysis,
    IRGraph,
    IRNode,
    ParallelismInfo,
    node_from_callable,
)

__all__ = [
    "IRGraph",
    "IRNode",
    "ParallelismInfo",
    "GraphParallelismAnalysis",
    "node_from_callable",
]
