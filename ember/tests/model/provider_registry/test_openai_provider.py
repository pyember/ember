import pytest
from typing import Any
from unittest.mock import patch, MagicMock

from src.ember.registry.model.provider_registry.openai.openai_provider import (
    OpenAIModel,
)
from src.ember.registry.model.schemas.cost import ModelCost, RateLimit
from src.ember.registry.model.schemas.model_info import ModelInfo
from src.ember.registry.model.schemas.provider_info import ProviderInfo
from src.ember.registry.model.schemas.chat_schemas import ChatRequest
from ember.src.ember.registry.model.model_registry_exceptions import InvalidPromptError


@pytest.fixture
def openai_model() -> OpenAIModel:
    """Creates an OpenAIModel instance for testing purposes."""
    provider_info: ProviderInfo = ProviderInfo(
        name="OpenAI", default_api_key="test_openai_key"
    )
    model_info: ModelInfo = ModelInfo(
        model_id="openai:gpt-4o",
        model_name="gpt-4o",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=provider_info,
        api_key="test_openai_key",
    )
    return OpenAIModel(model_info)


@patch("openai.resources.chat.Completions.create")
def test_forward_success(mock_create: Any, openai_model: OpenAIModel) -> None:
    """Verifies that a valid chat request is forwarded correctly and the response is processed."""
    # Set up the mock response for the chat.completions.create call.
    mock_response: MagicMock = MagicMock()
    mock_choice_message: MagicMock = MagicMock()
    mock_choice_message.content = "Hello from GPT-4 (mocked)"
    mock_choice: MagicMock = MagicMock()
    mock_choice.message = mock_choice_message
    mock_response.choices = [mock_choice]

    mock_usage: MagicMock = MagicMock()
    mock_usage.total_tokens = 42
    mock_usage.prompt_tokens = 21
    mock_usage.completion_tokens = 21
    mock_response.usage = mock_usage

    mock_create.return_value = mock_response

    # Create a ChatRequest using named parameters.
    chat_request: ChatRequest = ChatRequest(
        prompt="New test prompt", provider_params={"max_completion_tokens": 128}
    )

    # Use the explicitly named method 'forward' to invoke the API call.
    response = openai_model.forward(request=chat_request)

    # Assert that the response content and usage data match expectations.
    assert response.data == "Hello from GPT-4 (mocked)"
    assert response.usage.total_tokens == 42

    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs.get("model") == "gpt-4o"


@patch("openai.resources.chat.Completions.create")
def test_forward_empty_prompt(mock_create: Any, openai_model: OpenAIModel) -> None:
    """Checks that an empty prompt in a ChatRequest raises an InvalidPromptError."""
    # Construct a ChatRequest with an empty prompt.
    empty_request: ChatRequest = ChatRequest(prompt="", provider_params={})
    with pytest.raises(InvalidPromptError, match="OpenAI prompt cannot be empty."):
        openai_model.forward(request=empty_request)
