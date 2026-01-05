"""Tests for Anthropic provider parameter handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ember.models.providers.anthropic import AnthropicProvider


def _provider(*, require_stream: bool = False) -> tuple[AnthropicProvider, dict[str, object]]:
    state: dict[str, object] = {"stream_calls": 0}

    def _create(**params):
        state["create_params"] = params
        if require_stream:
            raise ValueError(
                "Streaming is strongly recommended for operations that may take longer than "
                "10 minutes."
            )
        usage = SimpleNamespace(input_tokens=0, output_tokens=0)
        return SimpleNamespace(content=[], usage=usage)

    def _stream(**params):
        state["stream_params"] = params
        state["stream_calls"] = int(state["stream_calls"]) + 1

        class _Stream:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def __iter__(self_inner):  # pragma: no cover
                return iter(())

            def get_final_response(self_inner):
                usage = SimpleNamespace(input_tokens=0, output_tokens=0)
                return SimpleNamespace(content=[], usage=usage)

        return _Stream()

    messages = SimpleNamespace(create=_create, stream=_stream)
    client = SimpleNamespace(messages=messages)

    provider = AnthropicProvider(api_key="test")
    provider.client = client  # type: ignore[assignment]
    provider._resolve_model_name = lambda model: model  # type: ignore[assignment]
    return provider, state


def test_max_output_tokens_prefers_response_field():
    provider, state = _provider()
    provider.complete("hello", "claude-3-opus", max_output_tokens=55)
    params = state.get("create_params")
    assert isinstance(params, dict)
    assert params["max_tokens"] == 55


def test_conflicting_max_token_fields_error():
    provider, _ = _provider()
    with pytest.raises(Exception) as exc:
        provider.complete("hi", "claude-3-opus", max_tokens=1, max_output_tokens=2)
    assert "max_tokens" in str(exc.value)


def test_streaming_fallback_invoked_on_sdk_value_error():
    provider, state = _provider(require_stream=True)
    response = provider.complete("hi", "claude-3-opus")
    assert response.text == ""
    assert state["stream_calls"] == 1
    assert "stream_params" in state
