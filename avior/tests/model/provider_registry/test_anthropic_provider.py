import pytest
from unittest.mock import patch, MagicMock
from avior.registry.model.provider_registry.anthropic.anthropic_provider import (
    AnthropicModel,
)
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from src.avior.registry.model.exceptions import InvalidPromptError


@pytest.fixture
def model_info():
    """
    A reusable fixture providing a ModelInfo object for tests.
    """
    from src.avior.registry.model.schemas.model_info import ModelInfo
    from src.avior.registry.model.schemas.provider_info import ProviderInfo
    from src.avior.registry.model.schemas.cost import ModelCost, RateLimit

    return ModelInfo(
        model_id="anthropic:claude-3.5-sonnet",
        model_name="claude-3.5-sonnet",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=ProviderInfo(name="Anthropic"),
        api_key="test_anthropic_key",
    )


@patch("anthropic.Client", autospec=True)
def test_create_client(mock_client_class, model_info):
    """
    Ensures anthropic.Client is instantiated with the correct API key.
    """
    from avior.registry.model.provider_registry.anthropic.anthropic_provider import (
        AnthropicModel,
    )

    anthro_model = AnthropicModel(model_info)

    mock_client_class.assert_called_once_with(api_key="test_anthropic_key")
    assert anthro_model.client is not None


@patch("anthropic.Client", autospec=True)
def test_forward_success(mock_client_class, model_info):
    """
    Verifies forward() uses the correct parameters and returns a valid response
    that can be parsed by Pydantic without error.
    """
    from avior.registry.model.provider_registry.anthropic.anthropic_provider import (
        AnthropicModel,
    )

    anthro_model = AnthropicModel(model_info)

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

    resp = anthro_model("New integration test", context="System context")
    assert resp.data == "Hello from Claude 3.5 Sonnet (mocked)"

    mock_client.completions.create.assert_called_once()
    _, call_kwargs = mock_client.completions.create.call_args
    assert call_kwargs["model"] == "claude-3.5-sonnet"
    assert "prompt" in call_kwargs
    assert "System context" in call_kwargs["prompt"]
    assert "New integration test" in call_kwargs["prompt"]


def test_forward_empty_prompt(model_info):
    """
    Attempting to forward an empty prompt should raise InvalidPromptError.
    """
    from avior.registry.model.provider_registry.anthropic.anthropic_provider import (
        AnthropicModel,
    )
    from src.avior.registry.model.exceptions import InvalidPromptError

    anthro_model = AnthropicModel(model_info)

    with pytest.raises(InvalidPromptError, match="Anthropic prompt cannot be empty."):
        anthro_model("")
