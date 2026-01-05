from __future__ import annotations

from typing import Any

from ember.models.schemas import UsageStats


def usage_from_openai(payload: Any) -> UsageStats | None:
    if payload is None:
        return None

    def _read_int(name: str) -> int:
        value = getattr(payload, name, None)
        if value is None and isinstance(payload, dict):
            value = payload.get(name)
        return int(value or 0)

    prompt_tokens = _read_int("prompt_tokens") or _read_int("input_tokens")
    completion_tokens = _read_int("completion_tokens") or _read_int("output_tokens")
    total_tokens = _read_int("total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    cost_raw = getattr(payload, "cost_usd", None)
    if cost_raw is None and isinstance(payload, dict):
        cost_raw = payload.get("cost_usd")

    actual_cost: float | None
    if cost_raw is None:
        actual_cost = None
    else:
        try:
            actual_cost = float(cost_raw)
        except (TypeError, ValueError):
            actual_cost = None

    detail_fields = (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "num_search_queries",
    )
    details: dict[str, int] = {}
    for field in detail_fields:
        raw_value = getattr(payload, field, None)
        if raw_value is None and isinstance(payload, dict):
            raw_value = payload.get(field)
        if raw_value is None:
            continue
        try:
            details[field] = int(raw_value)
        except (TypeError, ValueError):
            continue

    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        actual_cost_usd=actual_cost,
        details=details or None,
    )
