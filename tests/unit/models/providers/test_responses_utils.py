from ember.models.providers.utils import sanitize_responses_payload


def test_sanitize_responses_payload_removes_temperature_for_codex() -> None:
    payload = {"model": "gpt-5-codex", "temperature": 0.5, "top_p": 0.9}
    sanitized = sanitize_responses_payload("gpt-5-codex", payload)
    assert "temperature" not in sanitized
    assert "top_p" not in sanitized


def test_sanitize_responses_payload_retains_temperature_for_other_models() -> None:
    payload = {"model": "gpt-5", "temperature": 0.3}
    sanitized = sanitize_responses_payload("gpt-5", payload)
    assert sanitized.get("temperature") == 0.3


def test_sanitize_responses_payload_drops_reasoning_for_claude() -> None:
    payload = {"model": "claude-opus-4-1-20250805", "reasoning": {"effort": "high"}}
    sanitized = sanitize_responses_payload("claude-opus-4-1-20250805", payload)
    assert "reasoning" not in sanitized
