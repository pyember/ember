import logging
import anthropic
from typing import Any
import uuid
from tenacity import retry, wait_exponential, stop_after_attempt
from pydantic import Field, field_validator

from src.avior.registry.model.provider_registry.base import BaseProviderModel
from src.avior.registry.model.exceptions import InvalidPromptError, ProviderAPIError
from src.avior.registry.model.schemas.usage import UsageStats
from src.avior.registry.model.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    BaseChatParameters
)

logger = logging.getLogger(__name__)

class AnthropicChatParameters(BaseChatParameters):
    """
    For Anthropic, the param is 'max_tokens_to_sample'.
    Must be an integer. Default 768 if missing.
    """
    max_tokens: int | None = Field(default=None)

    @field_validator("max_tokens", mode="before")
    def enforce_default_if_none(cls, v):
        # If user sets None, we force 768
        return 768 if v is None else v

    @field_validator("max_tokens")
    def ensure_positive(cls, v):
        if v < 1:
            raise ValueError(f"max_tokens must be >= 1, got {v}")
        return v

    def to_anthropic_kwargs(self) -> dict:
        anthropic_prompt = f"{self.context or ''}\n\nHuman: {self.prompt}\n\nAssistant:"

        kwargs = {
            "prompt": anthropic_prompt,
            "max_tokens_to_sample": self.max_tokens,
        }

        # Only pass temperature if not None
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        return kwargs

class AnthropicModel(BaseProviderModel):
    """
    Concrete implementation for Anthropic models (Claude, etc.).
    """

    PROVIDER_NAME = "Anthropic"

    def _normalize_anthropic_model_name(self, raw_name: str) -> str:
        """
        If the specified model is not recognized by Anthropic,
        fallback to a known valid model (e.g., 'claude-instant-1').
        The official docs or error messages can guide which models are valid.
        """
        # For example, if 'claude-3.5-sonnet' is not recognized, we fallback:
        known_anthropic_models = {"claude-2", "claude-instant-1"}
        if raw_name not in known_anthropic_models:
            logger.warning(
                "Anthropic model '%s' not recognized by the API. Using 'claude-2'.",
                raw_name
            )
            return "claude-2"
        return raw_name

    def create_client(self) -> Any:
        api_key = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("Anthropic API key is missing or invalid.")
        return anthropic.Client(api_key=api_key)

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def forward(self, request: ChatRequest) -> ChatResponse:
        if not request.prompt:
            raise InvalidPromptError("Anthropic prompt cannot be empty.")

        correlation_id = request.provider_params.get("correlation_id") or str(uuid.uuid4())
        logger.info(
            "Anthropic forward() invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.model_name,
                "correlation_id": correlation_id,
                "prompt_length": len(request.prompt),
            },
        )

        # 0) Possibly fix the model name if it’s invalid
        final_model_name = self._normalize_anthropic_model_name(self.model_info.model_name)

        # 1) Convert universal ChatRequest -> AnthropicChatParameters
        anthropic_params = AnthropicChatParameters(**request.dict(exclude={"provider_params"}))
        anthro_kwargs = anthropic_params.to_anthropic_kwargs()

        # 2) Merge additional provider_params
        anthro_kwargs.update(request.provider_params)

        try:
            response = self.client.completions.create(
                model=final_model_name,
                **anthro_kwargs,
            )
            text = response.completion.strip()
            usage = self.calculate_usage(response)

            return ChatResponse(data=text, raw_output=response, usage=usage)

        except Exception as e:
            logger.exception("Anthropic model execution error.")
            raise ProviderAPIError(f"Error calling Anthropic: {e}") from e

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        """
        Extract approximate usage stats from the raw response.
        Adjust if real usage data is available.
        """
        completion_words = len(raw_output.completion.split())
        return UsageStats(
            total_tokens=completion_words,
            prompt_tokens=completion_words,
            completion_tokens=0,
            cost_usd=0.0,
        )
