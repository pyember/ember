import logging
import os
import pytest

from src.avior.registry.model.config import settings
from src.avior.registry.model.schemas.chat_schemas import ChatResponse
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.registry.model_registry import ModelRegistry
from src.avior.registry.model.services.usage_service import UsageService
from src.avior.registry.model.services.model_service import ModelService
from src.avior.registry.model.registry.model_enum import (
    AnthropicModelEnum,
    OpenAIModelEnum,
    GoogleModelEnum,
)

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_integration_anthropic_end_to_end():
    """
    Full integration test to ensure:
      1) We load/validate Anthropic API key.
      2) We register the model in our registry.
      3) We call the model in multiple ways: directly and via ModelService.
      4) We capture usage stats in UsageService and log them.
      5) We assert minimal expectations on the response to ensure it is non-empty.
    """

    # 1) Load Anthropic API Key (from environment or from Pydantic settings)
    anthropic_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        pytest.skip("Skipping test because ANTHROPIC_API_KEY is not set.")

    # 2) Create a ModelInfo for an Anthropic model
    anthro_model_info = ModelInfo(
        model_id=AnthropicModelEnum.CLAUDE_2.value,
        model_name="claude-2",
        cost=ModelCost(input_cost_per_thousand=0.0160, output_cost_per_thousand=0.0200),
        rate_limit=RateLimit(tokens_per_minute=90000, requests_per_minute=5000),
        provider=ProviderInfo(name="Anthropic", default_api_key=anthropic_key),
        api_key=anthropic_key,
    )

    # 3) Build registry/usage service/model service
    registry = ModelRegistry()
    usage_service = UsageService()
    model_service = ModelService(registry, usage_service)

    # 4) Register the Anthropic model
    registry.register_model(anthro_model_info)

    # 5a) Retrieve and call the model directly using get_model()
    direct_model = registry.get_model(anthro_model_info.model_id)
    assert direct_model is not None, "Registry failed to return Anthropic model."

    # Call the model directly
    direct_response: ChatResponse = direct_model("What is the capital of France?")
    logger.info("Direct model call response: %s", direct_response.data)
    assert direct_response.data, "Anthropic model returned empty data (direct call)."

    # 5b) Call the model via ModelService
    svc_response: ChatResponse = model_service(
        model_id=anthro_model_info.model_id,
        prompt="What's the capital city of Germany?",
    )
    logger.info("ModelService call response: %s", svc_response.data)
    assert svc_response.data, "Anthropic model returned empty data (service call)."

    # 6) Check usage service summary
    usage_summary = usage_service.get_usage_summary(anthro_model_info.model_id)
    logger.info("Usage summary after calls: %s", usage_summary.model_dump())
    assert (
        usage_summary.total_usage.total_tokens > 0
    ), "No tokens logged in usage summary."
    assert (
        usage_summary.total_usage.prompt_tokens > 0
    ), "Expected some prompt tokens to be recorded."

    logger.info("Integration test with Anthropic model completed successfully!")


@pytest.mark.integration
def test_integration_openai_end_to_end():
    """
    Integration test for OpenAI provider.
    Checks that we can register an OpenAI model, call it,
    and see usage stats > 0.
    """
    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("Skipping test because OPENAI_API_KEY is not set.")

    openai_model_info = ModelInfo(
        model_id=OpenAIModelEnum.GPT_4.value,
        model_name="gpt-4",
        cost=ModelCost(input_cost_per_thousand=0.03, output_cost_per_thousand=0.06),
        rate_limit=RateLimit(tokens_per_minute=80000, requests_per_minute=5000),
        provider=ProviderInfo(name="OpenAI", default_api_key=openai_key),
        api_key=openai_key,
    )

    registry = ModelRegistry()
    usage_service = UsageService()
    model_service = ModelService(registry, usage_service)
    registry.register_model(openai_model_info)

    direct_model = registry.get_model(openai_model_info.model_id)
    assert direct_model, "Registry failed to return OpenAI model."

    direct_response: ChatResponse = direct_model(
        "Hello from OpenAI test_integration!", max_completion_tokens=200
    )
    assert direct_response.data, "OpenAI model returned empty data (direct call)."

    svc_response: ChatResponse = model_service(
        model_id=openai_model_info.model_id,
        prompt="Another OpenAI call from test_integration.",
    )
    assert svc_response.data, "OpenAI model returned empty data (service call)."

    usage_summary = usage_service.get_usage_summary(openai_model_info.model_id)
    logger.info("OpenAI usage summary: %s", usage_summary.model_dump())
    assert (
        usage_summary.total_usage.total_tokens > 0
    ), "No tokens logged for OpenAI usage."


@pytest.mark.integration
def test_integration_gemini_end_to_end():
    """
    Integration test for Google Gemini (Generative AI).
    Similar to Anthropic and OpenAI tests.
    """
    google_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY")
    if not google_key:
        pytest.skip("Skipping test because GOOGLE_API_KEY is not set.")

    # Here we pick a known valid name, e.g. "models/gemini-1.5-flash"
    logger.info("Attempting a known valid Gemini model name.")
    gemini_model_info = ModelInfo(
        model_id=GoogleModelEnum.GEMINI_1_5_FLASH.value,
        model_name="gemini-1.5-flash",
        cost=ModelCost(input_cost_per_thousand=0.015, output_cost_per_thousand=0.02),
        rate_limit=RateLimit(tokens_per_minute=90000, requests_per_minute=5000),
        provider=ProviderInfo(name="Google", default_api_key=google_key),
        api_key=google_key,
    )

    registry = ModelRegistry()
    usage_service = UsageService()
    model_service = ModelService(registry, usage_service)
    registry.register_model(gemini_model_info)

    direct_model = registry.get_model(gemini_model_info.model_id)
    assert direct_model, "Registry failed to return Gemini model."

    direct_response: ChatResponse = direct_model(
        "Please summarize the US constitution."
    )
    assert direct_response.data, "Gemini model returned empty data (direct call)."

    svc_response: ChatResponse = model_service(
        gemini_model_info.model_id, "Explain quantum entanglement in simple terms."
    )
    assert svc_response.data, "Gemini model returned empty data (service call)."
    print('GEMINI SVC RESPONSE: ', svc_response)
    usage_summary = usage_service.get_usage_summary(gemini_model_info.model_id)
    logger.info("Gemini usage summary: %s", usage_summary.model_dump())
    assert (
        usage_summary.total_usage.total_tokens > 0
    ), "No tokens logged for Gemini usage."
