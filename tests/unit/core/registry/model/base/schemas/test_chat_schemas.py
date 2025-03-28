"""Unit tests for chat schemas (ChatRequest and ChatResponse).
Verifies that all fields validate and that JSON serialization works.
"""

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)


def test_chat_request_validation() -> None:
    """Test that a ChatRequest is correctly instantiated with all fields."""
    request = ChatRequest(
        prompt="Test prompt",
        context="Test context",
        max_tokens=100,
        temperature=0.5,
        provider_params={"extra": "value"},
    )
    assert request.prompt == "Test prompt"
    assert request.context == "Test context"
    assert request.max_tokens == 100
    assert request.temperature == 0.5
    assert request.provider_params == {"extra": "value"}


def test_chat_request_optional_fields() -> None:
    """Test ChatRequest with only the required field."""
    request = ChatRequest(prompt="Hello")
    assert request.context is None
    assert request.temperature is None
    assert request.max_tokens is None
    assert request.provider_params == {}


def test_chat_response_serialization() -> None:
    """Test that ChatResponse serializes and deserializes correctly."""
    response = ChatResponse(
        data="Response text", raw_output={"raw": "data"}, usage=None
    )
    json_str = response.model_dump_json()
    loaded = ChatResponse.model_validate_json(json_str)
    assert loaded.data == "Response text"
    assert loaded.raw_output == {"raw": "data"}
