import pytest
from unittest.mock import patch, MagicMock

from avior.registry.model.provider_registry.openai.openai_provider import OpenAIModel
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from src.avior.registry.model.exceptions import InvalidPromptError


@pytest.fixture
def openai_model():
    provider_info = ProviderInfo(name="OpenAI", default_api_key="test_openai_key")
    model_info = ModelInfo(
        model_id="openai:gpt-4o",
        model_name="gpt-4o",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=provider_info,
        api_key="test_openai_key",
    )
    return OpenAIModel(model_info)


@patch("openai.resources.chat.Completions.create")
def test_forward_success(mock_create, openai_model):
    mock_response = MagicMock()
    mock_choice_message = MagicMock()
    mock_choice_message.content = "Hello from GPT-4 (mocked)"
    mock_choice = MagicMock()
    mock_choice.message = mock_choice_message
    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.total_tokens = 42
    mock_usage.prompt_tokens = 21
    mock_usage.completion_tokens = 21
    mock_response.usage = mock_usage

    mock_create.return_value = mock_response

    resp = openai_model("New test prompt", max_completion_tokens=128)
    assert resp.data == "Hello from GPT-4 (mocked)"
    assert resp.usage.total_tokens == 42

    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs.get("model") == "gpt-4o"


@patch("openai.resources.chat.Completions.create")
def test_forward_empty_prompt(mock_create, openai_model):
    with pytest.raises(InvalidPromptError, match="OpenAI prompt cannot be empty."):
        openai_model("")
