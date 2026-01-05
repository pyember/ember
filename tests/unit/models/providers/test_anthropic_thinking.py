from ember.models.providers.anthropic.thinking import thinking_payload_from_reasoning


def test_thinking_payload_returns_none_for_empty_dict() -> None:
    """Empty dict has no effort or budget_tokens, so returns None."""
    payload = thinking_payload_from_reasoning({})
    assert payload is None


def test_thinking_payload_uses_effort_mapping() -> None:
    """Medium effort maps to 16384 tokens per implementation."""
    payload = thinking_payload_from_reasoning({"effort": "medium"})
    assert payload == {"type": "enabled", "budget_tokens": 16384}


def test_thinking_payload_accepts_numeric_budget() -> None:
    payload = thinking_payload_from_reasoning({"budget_tokens": 4096})
    assert payload == {"type": "enabled", "budget_tokens": 4096}


def test_thinking_payload_handles_disabled_flags() -> None:
    assert thinking_payload_from_reasoning(False) is None
    assert thinking_payload_from_reasoning({"enabled": False}) is None
