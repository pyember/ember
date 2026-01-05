"""Anthropic extended thinking payload translation.

Converts generic reasoning configurations into Anthropic's thinking API format.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

_EFFORT_TO_BUDGET_TOKENS: Final[dict[str, int]] = {
    "low": 4096,
    "medium": 16384,
    "high": 65536,
}


def thinking_payload_from_reasoning(
    reasoning: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert a reasoning configuration into an Anthropic thinking payload.

    Translates the generic reasoning abstraction (with effort levels) into
    Anthropic's extended thinking format which requires explicit token budgets.

    Args:
        reasoning: Reasoning configuration, typically containing an 'effort'
            key with values 'low', 'medium', or 'high'. May also contain
            'budget_tokens' for explicit token budget control.

    Returns:
        Anthropic thinking payload with 'type' and 'budget_tokens' keys,
        or None if reasoning is disabled or not provided.

    Examples:
        >>> thinking_payload_from_reasoning({"effort": "high"})
        {'type': 'enabled', 'budget_tokens': 65536}
        >>> thinking_payload_from_reasoning({"budget_tokens": 10000})
        {'type': 'enabled', 'budget_tokens': 10000}
        >>> thinking_payload_from_reasoning(None) is None
        True
    """
    if reasoning is None:
        return None

    if not isinstance(reasoning, Mapping):
        return None

    explicit_budget = reasoning.get("budget_tokens")
    if explicit_budget is not None:
        try:
            budget = int(explicit_budget)
        except (TypeError, ValueError):
            return None
        return {"type": "enabled", "budget_tokens": max(1024, budget)}

    effort = reasoning.get("effort")
    if effort is None:
        return None

    effort_key = str(effort).lower().strip()
    budget_tokens = _EFFORT_TO_BUDGET_TOKENS.get(effort_key)
    if budget_tokens is None:
        return None

    return {"type": "enabled", "budget_tokens": budget_tokens}
