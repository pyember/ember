"""Profiler helpers for XCS."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ember.xcs.api.state import PROFILER


def get_stats(func: Optional[str] = None) -> Dict[str, Any]:
    """Return profiler statistics for `func` or all functions."""
    if func is None:
        return PROFILER.summary()
    return PROFILER.get(func)


def reset_stats() -> None:
    """Clear all recorded profiling data."""
    PROFILER.clear()


__all__ = ["get_stats", "reset_stats"]
