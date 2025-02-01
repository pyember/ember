import pytest
from unittest.mock import patch, MagicMock
from avior.registry.model.provider_registry.deepmind.gemini_provider import GeminiModel
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from src.avior.registry.model.exceptions import InvalidPromptError


@patch("google.generativeai.configure")
def test_create_client(mock_configure):
    """
    Ensure google.generativeai.configure() is called with correct API key.
    Patch is applied first, then build GeminiModel so the patch sees the call.
    """
    test_info = ModelInfo(
        model_id="google:gemini-1.5-flash",
        model_name="gemini-1.5-flash",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=ProviderInfo(name="Google", default_api_key="test_google_key"),
        api_key="test_google_key",
    )
    gmodel = GeminiModel(test_info)  # creation calls create_client
    mock_configure.assert_called_once_with(api_key="test_google_key")


@patch(
    "google.generativeai.list_models",
    return_value=[
        type(
            "obj",
            (),
            {
                "name": "models/gemini-1.5-flash",
                "supported_generation_methods": ["generateContent"],
            },
        )
    ],
)
@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
def test_forward_success_new_model(mock_gen_model, mock_configure, mock_list):
    """
    Mocks the Gemini model retrieval and generate_content method.
    """
    test_info = ModelInfo(
        model_id="google:gemini-1.5-flash",
        model_name="gemini-1.5-flash",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=ProviderInfo(name="Google", default_api_key="test_google_key"),
        api_key="test_google_key",
    )
    gemini_model = GeminiModel(test_info)

    mock_model_instance = MagicMock()
    usage_mock = MagicMock()
    usage_mock.prompt_token_count = 10
    usage_mock.candidates_token_count = 20
    usage_mock.total_token_count = 30

    generated_mock = MagicMock()
    generated_mock.candidates = [
        MagicMock(content="Hello from Gemini 1.5 Flash (mocked)")
    ]
    generated_mock.usage_metadata = usage_mock

    mock_model_instance.generate_content.return_value = generated_mock
    mock_gen_model.return_value = mock_model_instance

    resp = gemini_model("Updated test prompt", context="System context")
    assert resp.data != None


def test_forward_empty_prompt():
    """
    Checks that empty prompts raise InvalidPromptError, no patch needed.
    """
    test_info = ModelInfo(
        model_id="google:gemini-1.5-flash",
        model_name="gemini-1.5-flash",
        cost=ModelCost(),
        rate_limit=RateLimit(),
        provider=ProviderInfo(name="Google", default_api_key="test_google_key"),
        api_key="test_google_key",
    )
    gemini_model = GeminiModel(test_info)
    with pytest.raises(InvalidPromptError, match="Gemini prompt cannot be empty."):
        gemini_model("")
