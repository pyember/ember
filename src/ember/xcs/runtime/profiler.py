"""Lightweight profiling support for XCS."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Mapping, Optional

from ember.xcs.compiler.graph import GraphParallelismAnalysis


@dataclass(slots=True)
class FunctionStats:
    """Aggregated timing information for a single function."""

    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    avg_speedup: float = 1.0

    def record(self, elapsed_ms: float, speedup: float) -> None:
        self.call_count += 1
        self.total_time_ms += elapsed_ms
        self.min_time_ms = min(self.min_time_ms, elapsed_ms)
        self.max_time_ms = max(self.max_time_ms, elapsed_ms)
        if self.call_count == 1:
            self.avg_speedup = speedup
        else:
            self.avg_speedup = (
                (self.avg_speedup * (self.call_count - 1)) + speedup
            ) / self.call_count

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.call_count if self.call_count else 0.0


class Profiler:
    """Collect and report profiling data for jitted functions."""

    def __init__(self) -> None:
        self._stats: DefaultDict[str, FunctionStats] = defaultdict(
            lambda: FunctionStats(name="unknown")
        )

    def record(
        self,
        func_name: str,
        elapsed_ms: float,
        parallelism: Optional[GraphParallelismAnalysis] = None,
    ) -> None:
        stats = self._stats[func_name]
        if stats.name == "unknown":
            stats.name = func_name
        speedup = parallelism.estimated_speedup if parallelism else 1.0
        stats.record(elapsed_ms, speedup)

    def get(self, func_name: str) -> Mapping[str, float]:
        stats = self._stats.get(func_name)
        if not stats:
            return {}
        return {
            "calls": float(stats.call_count),
            "total_time_ms": stats.total_time_ms,
            "avg_time_ms": stats.avg_time_ms,
            "min_time_ms": stats.min_time_ms,
            "max_time_ms": stats.max_time_ms,
            "avg_speedup": stats.avg_speedup,
        }

    def summary(self) -> Mapping[str, Mapping[str, float]]:
        return {name: self.get(name) for name in self._stats}

    def clear(self) -> None:
        self._stats.clear()


__all__ = ["Profiler", "FunctionStats"]
