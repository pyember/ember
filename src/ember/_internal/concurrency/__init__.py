"""Adaptive concurrency control for LLM API calls.

This module provides a feedback-driven concurrency controller that discovers
effective limits by observing outcomes (success, rate-limit, error). It uses
the AIMD (Additive Increase, Multiplicative Decrease) algorithm from TCP
congestion control.

Key concepts:
- **ConcurrencyKey**: Determines the isolation boundary for rate limits.
  Requests with different keys have independent limits.
- **Outcome**: The result of a provider call (SUCCESS, RATE_LIMITED, ERROR).
- **AdaptiveConcurrencyLimiter**: The core AIMD algorithm.

Independence is emergent from keying: by default, we key by provider name
(e.g., "openai", "anthropic"), which matches how most providers enforce limits.

Example usage:
    from ember._internal.concurrency import (
        AdaptiveConcurrencyLimiter,
        ConcurrencyKey,
        Outcome,
    )

    limiter = AdaptiveConcurrencyLimiter()
    key = ConcurrencyKey(provider="openai")

    acquired = limiter.acquire(key, timeout_s=30.0)
    if not acquired:
        raise TimeoutError("Concurrency limit timeout")

    try:
        response = call_provider(...)
        limiter.release(key, Outcome.SUCCESS)
    except RateLimitError:
        limiter.release(key, Outcome.RATE_LIMITED)
        raise

References:
    - Netflix concurrency-limits: https://github.com/Netflix/concurrency-limits
    - TCP Congestion Control (RFC 5681): https://datatracker.ietf.org/doc/html/rfc5681
"""

from ember._internal.concurrency.limiter import (
    AdaptiveConcurrencyLimiter,
    AdjustmentCallback,
    ConcurrencyKey,
    Outcome,
)

__all__ = [
    "AdaptiveConcurrencyLimiter",
    "AdjustmentCallback",
    "ConcurrencyKey",
    "Outcome",
]
