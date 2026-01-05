"""Core AIMD-based adaptive concurrency limiter.

This module contains the pure algorithm implementation without any dependencies
on gateway, metrics, or commercial code. It can be used by:
- models/ layer for per-provider rate limiting
- operators/ layer for parallel ensemble execution
- gateway/ for commercial multi-tenant rate limiting

The algorithm is inspired by TCP congestion control (RFC 5681) and Netflix's
concurrency-limits library. Key principles:

1. Start with an initial concurrency limit (default: 10)
2. On success: slowly increase limit (additive increase)
3. On 429/rate-limit: rapidly decrease limit (multiplicative decrease)
4. On other errors: no change (avoid over-reacting to transients)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

_LOG = logging.getLogger(__name__)


class Outcome(Enum):
    """Result of a provider call, used to adjust concurrency.

    The outcome determines how the concurrency limit is adjusted:
    - SUCCESS: Additive increase (slowly probe for more capacity)
    - RATE_LIMITED: Multiplicative decrease (rapidly back off)
    - ERROR: No change (avoid over-reacting to transient failures)
    - TIMEOUT: No change (could be network, not provider capacity)
    """

    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class ConcurrencyKey:
    """Key for concurrency isolation.

    Independence is determined by key equality. Two keys that are equal share
    concurrency limits; unequal keys are independent.

    By default, we key by provider only. This matches most provider rate limit
    boundaries (OpenAI, Anthropic, Google all enforce limits at the org level,
    not per-model).

    Attributes:
        provider: The provider name (e.g., "openai", "anthropic").
        model: Optional model name for per-model keying.
        org_id: Optional org ID for multi-tenant scenarios.
    """

    provider: str
    model: str | None = None
    org_id: str | None = None

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        *,
        include_model: bool = False,
        org_id: str | None = None,
    ) -> ConcurrencyKey:
        """Create a key from a model ID like 'openai/gpt-4o' or 'gpt-4o'.

        Args:
            model_id: Model identifier, optionally prefixed with provider.
            include_model: If True, include the model name in the key.
                          Default is False (key by provider only).
            org_id: Optional organization ID for multi-tenant isolation.

        Returns:
            ConcurrencyKey for the model.
        """
        # Import here to avoid circular dependency
        from ember.models.providers import resolve_model_id

        provider, model_name = resolve_model_id(model_id)
        return cls(
            provider=provider,
            model=model_name if include_model else None,
            org_id=org_id,
        )

    def __str__(self) -> str:
        """Human-readable key representation."""
        parts = [self.provider]
        if self.model:
            parts.append(self.model)
        if self.org_id:
            parts.append(self.org_id)
        return "/".join(parts)


@dataclass
class _ConcurrencyState:
    """Per-key concurrency state. Thread-safe access via external lock."""

    limit: float
    inflight: int = 0
    total_requests: int = 0
    total_429s: int = 0
    last_adjustment_time: float = field(default_factory=time.monotonic)


# Type alias for the adjustment callback
AdjustmentCallback = Callable[[ConcurrencyKey, float, float, Outcome], None]


class AdaptiveConcurrencyLimiter:
    """AIMD-based adaptive concurrency limiter.

    This limiter discovers effective concurrency limits through feedback:
    - On success: limit += additive_increase
    - On 429: limit = max(min_limit, limit * multiplicative_decrease)
    - On error/timeout: no change (transient, avoid over-reaction)

    Keys determine independence domains. Different keys adapt independently.
    By default, we recommend keying by provider name.

    Thread-safe: Uses threading.Condition for blocking acquire and safe
    concurrent access to shared state.

    Attributes:
        initial_limit: Starting concurrency limit for new keys.
        min_limit: Floor for concurrency limit (never go below this).
        max_limit: Ceiling for concurrency limit (never exceed this).
        additive_increase: Amount to increase limit per success.
        multiplicative_decrease: Factor to multiply limit on 429.
        cooldown_after_429_s: Seconds to wait before increasing after 429.
    """

    def __init__(
        self,
        initial_limit: float = 10.0,
        min_limit: float = 1.0,
        max_limit: float = 100.0,
        additive_increase: float = 0.1,
        multiplicative_decrease: float = 0.5,
        cooldown_after_429_s: float = 1.0,
    ) -> None:
        """Initialize the limiter with AIMD parameters.

        Args:
            initial_limit: Starting concurrency limit for new keys. Default 10.
            min_limit: Minimum concurrency limit. Default 1.
            max_limit: Maximum concurrency limit. Default 100.
            additive_increase: Amount to add per success. Default 0.1
                              (10 successes = +1 concurrency slot).
            multiplicative_decrease: Factor on 429 (e.g., 0.5 = halve). Default 0.5.
            cooldown_after_429_s: Seconds after 429 before allowing increase.
                                  Default 1.0. Set to 0 to disable.
        """
        self._lock = threading.Lock()
        self._states: dict[ConcurrencyKey, _ConcurrencyState] = {}
        self._condition = threading.Condition(self._lock)

        self.initial_limit = float(initial_limit)
        self.min_limit = float(min_limit)
        self.max_limit = float(max_limit)
        self.additive_increase = float(additive_increase)
        self.multiplicative_decrease = float(multiplicative_decrease)
        self.cooldown_after_429_s = float(cooldown_after_429_s)

        self._on_adjustment: AdjustmentCallback | None = None

    def set_on_adjustment(self, callback: AdjustmentCallback | None) -> None:
        """Set a callback to be invoked on limit adjustments.

        Useful for observability (logging, metrics).

        Args:
            callback: Function taking (key, old_limit, new_limit, outcome).
                     Pass None to clear.
        """
        self._on_adjustment = callback

    def _get_state(self, key: ConcurrencyKey) -> _ConcurrencyState:
        """Get or create state for key. Must hold lock."""
        if key not in self._states:
            self._states[key] = _ConcurrencyState(limit=self.initial_limit)
        return self._states[key]

    def acquire(self, key: ConcurrencyKey, timeout_s: float = 30.0) -> bool:
        """Acquire a concurrency slot for the given key.

        Blocks until a slot is available or timeout expires.

        Args:
            key: The concurrency key (usually provider-based).
            timeout_s: Maximum seconds to wait. Default 30.

        Returns:
            True if slot acquired, False if timeout.
        """
        deadline = time.monotonic() + timeout_s

        with self._condition:
            while True:
                state = self._get_state(key)
                effective_limit = max(1, int(state.limit))

                if state.inflight < effective_limit:
                    state.inflight += 1
                    state.total_requests += 1
                    _LOG.debug(
                        "Acquired slot for %s: inflight=%d, limit=%.1f",
                        key,
                        state.inflight,
                        state.limit,
                    )
                    return True

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _LOG.warning(
                        "Acquire timeout for %s: inflight=%d, limit=%.1f",
                        key,
                        state.inflight,
                        state.limit,
                    )
                    return False

                # Wait for a release, but don't wait forever
                self._condition.wait(timeout=min(remaining, 1.0))

    def release(self, key: ConcurrencyKey, outcome: Outcome) -> None:
        """Release a slot and adjust limit based on outcome.

        Args:
            key: The concurrency key used in acquire().
            outcome: The result of the operation (SUCCESS, RATE_LIMITED, etc.).
        """
        with self._condition:
            state = self._states.get(key)
            if state is None:
                _LOG.warning("Release called for unknown key: %s", key)
                return

            state.inflight = max(0, state.inflight - 1)
            old_limit = state.limit
            now = time.monotonic()

            if outcome == Outcome.RATE_LIMITED:
                # Multiplicative decrease: halve (or by configured factor)
                state.limit = max(self.min_limit, state.limit * self.multiplicative_decrease)
                state.total_429s += 1
                state.last_adjustment_time = now
                _LOG.info(
                    "Rate limited on %s: limit %.1f -> %.1f (total 429s: %d)",
                    key,
                    old_limit,
                    state.limit,
                    state.total_429s,
                )

            elif outcome == Outcome.SUCCESS:
                # Additive increase, but respect cooldown after 429
                if now - state.last_adjustment_time > self.cooldown_after_429_s:
                    state.limit = min(self.max_limit, state.limit + self.additive_increase)
                    if old_limit != state.limit:
                        _LOG.debug(
                            "Success on %s: limit %.1f -> %.1f",
                            key,
                            old_limit,
                            state.limit,
                        )

            # ERROR and TIMEOUT: no adjustment (transient)
            elif outcome in (Outcome.ERROR, Outcome.TIMEOUT):
                _LOG.debug("Non-429 outcome on %s: %s (no limit change)", key, outcome)

            # Invoke callback if limit changed
            if self._on_adjustment and old_limit != state.limit:
                try:
                    self._on_adjustment(key, old_limit, state.limit, outcome)
                except Exception:
                    _LOG.exception("Adjustment callback failed")

            # Wake up any waiters
            self._condition.notify_all()

    def get_stats(self, key: ConcurrencyKey) -> dict[str, float | int]:
        """Get current stats for a key.

        Useful for observability and debugging.

        Args:
            key: The concurrency key.

        Returns:
            Dict with limit, inflight, total_requests, total_429s.
        """
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return {
                    "limit": self.initial_limit,
                    "inflight": 0,
                    "total_requests": 0,
                    "total_429s": 0,
                }
            return {
                "limit": state.limit,
                "inflight": state.inflight,
                "total_requests": state.total_requests,
                "total_429s": state.total_429s,
            }

    def get_all_stats(self) -> dict[str, dict[str, float | int]]:
        """Get stats for all tracked keys.

        Returns:
            Dict mapping key string to stats dict.
        """
        with self._lock:
            return {str(k): self.get_stats(k) for k in self._states}


# Global singleton for cross-request state persistence
_global_limiter: AdaptiveConcurrencyLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> AdaptiveConcurrencyLimiter:
    """Get the global adaptive concurrency limiter.

    The limiter is a singleton to maintain state across requests.
    State persists for the lifetime of the process.

    Returns:
        The global AdaptiveConcurrencyLimiter instance.
    """
    global _global_limiter
    with _limiter_lock:
        if _global_limiter is None:
            _global_limiter = AdaptiveConcurrencyLimiter()
            _LOG.info("Initialized global adaptive concurrency limiter")
        return _global_limiter


def reset_limiter() -> None:
    """Reset the global limiter. For testing only."""
    global _global_limiter
    with _limiter_lock:
        _global_limiter = None


def outcome_from_error(exc: BaseException) -> Outcome:
    """Map an exception to an Outcome for the limiter.

    Args:
        exc: The exception raised by provider call.

    Returns:
        Outcome.RATE_LIMITED if it's a 429, otherwise Outcome.ERROR.
    """
    from ember._internal.exceptions import ProviderAPIError

    if isinstance(exc, ProviderAPIError):
        context = exc.context or {}
        if context.get("error_type") == "rate_limit":
            return Outcome.RATE_LIMITED
    return Outcome.ERROR
