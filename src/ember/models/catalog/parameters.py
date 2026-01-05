"""Declarative parameter policies for canonical models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResponsesParamPolicy:
    """Controls which sampling knobs Ember may pass to a model."""

    allow_temperature: bool = True
    allow_top_p: bool = True
    allow_reasoning: bool = True


_DEFAULT_POLICY = ResponsesParamPolicy()
_POLICY_BY_MODEL: dict[str, ResponsesParamPolicy] = {
    # Codex and nano endpoints reject sampling params such as temperature.
    "gpt-5-codex": ResponsesParamPolicy(allow_temperature=False, allow_top_p=False),
    "gpt-5-nano": ResponsesParamPolicy(allow_temperature=False, allow_top_p=False),
    "gpt-5-nano-2025-08-07": ResponsesParamPolicy(allow_temperature=False, allow_top_p=False),
    # Anthropic Responses does not support reasoning controls yet.
    "claude-opus-4-1-20250805": ResponsesParamPolicy(allow_reasoning=False),
    "claude-sonnet-4-5-20250929": ResponsesParamPolicy(allow_reasoning=False),
}


def get_responses_policy(model_id: str) -> ResponsesParamPolicy:
    """Return the parameter policy for ``model_id``."""

    key = model_id.strip().lower()
    return _POLICY_BY_MODEL.get(key, _DEFAULT_POLICY)


__all__ = ["ResponsesParamPolicy", "get_responses_policy"]
