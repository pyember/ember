"""Backward-compatible parallelism analyzer alias."""

from ember.xcs.compiler.graph import GraphParallelismAnalysis, ParallelismInfo
from ember.xcs.compiler.parallelism import ParallelismAnalyzer

__all__ = ["ParallelismAnalyzer", "ParallelismInfo", "GraphParallelismAnalysis"]
