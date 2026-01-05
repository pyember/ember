"""Lightweight metrics helpers for integrations.

This module provides a small, opinionated ``MetricsContext`` that powers
observability for the optional integrations (MCP, DSPy, Swarm).  The context
records latency, success/failure state, and token usage for the most recent
span while also aggregating coarse statistics that can be surfaced to users.

The implementation intentionally keeps state in-process and side-effect free so
it can back tests and local development without requiring Prometheus or other
external collectors.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Literal, Optional

def _coerce_usage(payload: Any) -> Dict[str, int]:
    """Convert different payload shapes into a normalized usage dictionary."""

    if payload is None:
        return {}

    data: Dict[str, Any]

    if hasattr(payload, "model_dump"):
        data = payload.model_dump()  # type: ignore[assignment]
    elif hasattr(payload, "dict"):
        data = payload.dict()  # type: ignore[assignment]
    elif isinstance(payload, dict):
        data = payload
    else:
        return {}

    usage: Dict[str, int] = {}

    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = data.get(key)
        if value is not None:
            try:
                usage[key] = int(value)
            except (TypeError, ValueError):
                continue

    # Ensure total_tokens is always present when both prompt/completion exist
    if "total_tokens" not in usage:
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = prompt + completion
        if total:
            usage["total_tokens"] = total

    return usage


def _extract_usage(response: Any) -> Dict[str, int]:
    """Best-effort extraction of token usage from a model response."""

    if response is None:
        return {}

    if hasattr(response, "usage"):
        usage = _coerce_usage(response.usage)  # type: ignore[attr-defined]
        if usage:
            return usage

    if isinstance(response, dict) and "usage" in response:
        usage = _coerce_usage(response["usage"])
        if usage:
            return usage

    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, dict) and "usage" in metadata:
        usage = _coerce_usage(metadata["usage"])
        if usage:
            return usage

    return {}


def _percentile(values: deque[float], fraction: float) -> float:
    """Compute a percentile from a bounded deque."""

    if not values:
        return 0.0

    ordered = sorted(values)
    idx = int(len(ordered) * fraction)
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


@dataclass
class _MetricsTotals:
    total_calls: int = 0
    total_success: int = 0
    total_failure: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0


class MetricsContext:
    """Collect latency and usage metrics for model integrations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latencies: deque[float] = deque(maxlen=512)
        self._totals = _MetricsTotals()
        self._last_metrics: Dict[str, Any] = {}

    def track(self, *, model: Optional[str] = None) -> "_MetricsSpan":
        """Return a span that records metrics for a single model invocation."""

        return _MetricsSpan(context=self, model=model)

    def get_last_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of the most recent span's metrics."""

        with self._lock:
            return dict(self._last_metrics)

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Return aggregate statistics across all recorded spans."""

        with self._lock:
            calls = self._totals.total_calls
            success = self._totals.total_success
            latencies = list(self._latencies)
            avg_latency = mean(latencies) if latencies else 0.0
            p50 = _percentile(self._latencies, 0.5)
            p95 = _percentile(self._latencies, 0.95)
            success_rate = (success / calls * 100.0) if calls else 0.0

            return {
                "total_calls": calls,
                "success": success,
                "failure": self._totals.total_failure,
                "success_rate_pct": success_rate,
                "latency_ms_avg": avg_latency,
                "latency_ms_p50": p50,
                "latency_ms_p95": p95,
                "prompt_tokens": self._totals.prompt_tokens,
                "completion_tokens": self._totals.completion_tokens,
                "total_tokens": self._totals.total_tokens,
                "total_cost": self._totals.total_cost,
            }

    def reset(self) -> None:
        """Clear all collected metrics."""

        with self._lock:
            self._latencies.clear()
            self._totals = _MetricsTotals()
            self._last_metrics.clear()

    # Internal helpers -------------------------------------------------

    def _finalize_span(
        self,
        *,
        success: bool,
        latency_ms: float,
        usage: Dict[str, int],
        model: Optional[str],
        metadata: Dict[str, Any],
        error: Optional[str],
    ) -> None:
        with self._lock:
            self._totals.total_calls += 1
            if success:
                self._totals.total_success += 1
            else:
                self._totals.total_failure += 1

            self._latencies.append(latency_ms)

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            self._totals.prompt_tokens += prompt_tokens
            self._totals.completion_tokens += completion_tokens
            self._totals.total_tokens += total_tokens

            cost = metadata.get("cost")
            if isinstance(cost, (int, float)):
                self._totals.total_cost += float(cost)

            last_metrics = {
                "success": success,
                "latency_ms": latency_ms,
                "usage": usage,
                "metadata": metadata,
            }
            if model:
                last_metrics["model"] = model
            if error:
                last_metrics["error"] = error

            self._last_metrics = last_metrics


@dataclass
class _MetricsSpan:
    """Context manager that records a single invocation's metrics."""

    context: MetricsContext
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._started_at = 0.0
        self._usage: Dict[str, int] = {}
        self._success = True
        self._error: Optional[str] = None

    def __enter__(self) -> "_MetricsSpan":
        self._started_at = time.perf_counter()
        return self

    def record_response(
        self,
        response: Any,
        *,
        usage: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        model: Optional[str] = None,
    ) -> None:
        """Record response metadata while inside the span."""

        if model:
            self.model = model

        if usage is None:
            usage = _extract_usage(response)

        self._usage = usage or {}

        if cost is not None:
            self.metadata["cost"] = float(cost)

    def record_usage(
        self,
        *,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Explicitly set usage details when automatic extraction is insufficient."""

        usage: Dict[str, int] = {}
        if prompt_tokens is not None:
            usage["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage["completion_tokens"] = int(completion_tokens)
        if total_tokens is not None:
            usage["total_tokens"] = int(total_tokens)

        if "total_tokens" not in usage and usage:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            usage["total_tokens"] = prompt + completion

        self._usage = usage

    def add_metadata(self, **metadata: Any) -> None:
        """Attach additional metadata to the span."""

        self.metadata.update(metadata)

    def record_failure(self, error_message: str) -> None:
        """Mark the span as failed with an error message."""

        self._success = False
        self._error = error_message

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        latency_ms = (time.perf_counter() - self._started_at) * 1000.0

        success = self._success and exc_type is None
        error_message = self._error

        if exc_type is not None:
            success = False
            error_message = str(exc)

        self.context._finalize_span(
            success=success,
            latency_ms=latency_ms,
            usage=self._usage,
            model=self.model,
            metadata=dict(self.metadata),
            error=error_message,
        )

        # Do not swallow exceptions
        return False


__all__ = ["MetricsContext"]
