"""Compiler utilities for XCS."""

from ember.xcs.compiler.analysis import OperationSummary, analyze_operations
from ember.xcs.compiler.builder import IRBuilder
from ember.xcs.compiler.graph import (
    GraphParallelismAnalysis,
    IRGraph,
    IRNode,
    ParallelismInfo,
    node_from_callable,
)
from ember.xcs.compiler.parallelism import ParallelismAnalyzer

__all__ = [
    "IRBuilder",
    "IRGraph",
    "IRNode",
    "GraphParallelismAnalysis",
    "ParallelismInfo",
    "ParallelismAnalyzer",
    "OperationSummary",
    "analyze_operations",
    "node_from_callable",
]
