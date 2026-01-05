"""Shared singleton instances used across XCS public APIs."""

from ember.xcs.compiler.builder import IRBuilder
from ember.xcs.compiler.parallelism import ParallelismAnalyzer
from ember.xcs.runtime.engine import ExecutionEngine
from ember.xcs.runtime.profiler import Profiler

ENGINE = ExecutionEngine()
PROFILER = Profiler()
BUILDER = IRBuilder()
ANALYZER = ParallelismAnalyzer()

__all__ = ["ENGINE", "PROFILER", "BUILDER", "ANALYZER"]
