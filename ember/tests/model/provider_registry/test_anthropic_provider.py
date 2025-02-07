import pytest
from typing import Any
from unittest.mock import patch, MagicMock

from src.ember.registry.model.provider_registry.anthropic.anthropic_provider import (
    AnthropicModel,
)
from src.ember.registry.model.schemas.model_info import ModelInfo
from src.ember.registry.model.schemas.provider_info import ProviderInfo
from src.ember.registry.model.schemas.cost import ModelCost, RateLimit
from src.ember.registry.model.schemas.chat_schemas import ChatRequest
from ember.src.ember.registry.model.model_registry_exceptions import InvalidPromptError


@pytest.fixture
def model_info() -> ModelInfo:
    """Creates and returns a ModelInfo instance for Anthropic provider tests.

    Returns:
        ModelInfo: A configured model info instance with Anthropic settings.
    """
    return ModelInfo(
        model_id="anthropic:claude-3.5-sonnet",
        model_name="claude-3.5-sonnet",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=ProviderInfo(name="Anthropic"),
        api_key="test_anthropic_key",
    )


@patch("anthropic.Client", autospec=True)
def test_create_client(mock_client_class: Any, model_info: ModelInfo) -> None:
    """Verifies that anthropic.Client is instantiated with the correct API key.

    Args:
        mock_client_class (Any): The patched anthropic.Client class.
        model_info (ModelInfo): The model info fixture.
    """
    anthropic_model: AnthropicModel = AnthropicModel(model_info=model_info)
    mock_client_class.assert_called_once_with(api_key="test_anthropic_key")
    assert anthropic_model.client is not None


@patch("anthropic.Client", autospec=True)
def test_forward_success(mock_client_class: Any, model_info: ModelInfo) -> None:
    """Checks that a successful chat request returns a valid response with proper usage stats.

    This test ensures that forward() generates the correct parameters and that the response,
    including usage information, is parsed correctly.

    Args:
        mock_client_class (Any): The patched anthropic.Client class.
        model_info (ModelInfo): The model info fixture.
    """
    anthropic_model: AnthropicModel = AnthropicModel(model_info=model_info)

    # Set up the mocked client and response.
    mock_client = mock_client_class.return_value
    mock_client.completions = MagicMock()

    mock_response = MagicMock()
    mock_response.completion = "Hello from Claude 3.5 Sonnet (mocked)"
    mock_usage = MagicMock()
    mock_usage.total_tokens = 40
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 25
    mock_response.usage = mock_usage
    mock_client.completions.create.return_value = mock_response

    # Create a ChatRequest with both prompt and context.
    chat_request: ChatRequest = ChatRequest(
        prompt="New integration test", context="System context"
    )
    response = anthropic_model.forward(request=chat_request)
    assert response.data == "Hello from Claude 3.5 Sonnet (mocked)"

    mock_client.completions.create.assert_called_once()
    _, call_kwargs = mock_client.completions.create.call_args
    assert call_kwargs.get("model") == "claude-3.5-sonnet"
    assert "prompt" in call_kwargs
    prompt_value: str = call_kwargs["prompt"]
    assert "System context" in prompt_value
    assert "New integration test" in prompt_value


def test_forward_empty_prompt(model_info: ModelInfo) -> None:
    """Ensures that an empty prompt results in an InvalidPromptError.

    Args:
        model_info (ModelInfo): The model info fixture.
    """
    anthropic_model: AnthropicModel = AnthropicModel(model_info=model_info)
    empty_chat_request: ChatRequest = ChatRequest(prompt="", context="")
    with pytest.raises(InvalidPromptError, match="Anthropic prompt cannot be empty."):
        anthropic_model.forward(request=empty_chat_request)
