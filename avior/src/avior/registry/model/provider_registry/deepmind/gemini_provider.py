import logging
from typing import Any

import google.generativeai as genai
from google.generativeai import GenerativeModel, types
from google.api_core.exceptions import NotFound
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

class GeminiChatParameters(BaseChatParameters):
    """
    For Google Gemini, we place 'max_tokens' into the 'max_output_tokens'
    field within a GenerationConfig object.
    """
    max_tokens: int | None = Field(default=None)

    @field_validator("max_tokens", mode="before")
    def enforce_default_if_none(cls, v):
        # If user sets None, we force it to 512
        return 512 if v is None else v

    @field_validator("max_tokens")
    def ensure_positive(cls, v):
        if v < 1:
            raise ValueError(f"max_tokens must be >= 1, got {v}")
        return v

    def to_gemini_kwargs(self) -> dict:
        generation_cfg = {
            "max_output_tokens": self.max_tokens,
        }
        # {{ Only pass temperature if not None }}
        if self.temperature is not None:
            generation_cfg["temperature"] = self.temperature

        return {"generation_config": generation_cfg}

class GeminiModel(BaseProviderModel):
    """
    Google Gemini provider implementation.
    """

    PROVIDER_NAME = "Google"

    def create_client(self) -> Any:
        """
        Configures the google.generativeai SDK with the appropriate API key.
        Logs available Gemini models for insight/debugging.
        """
        api_key = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("Google API key is missing or invalid.")
        genai.configure(api_key=api_key)

        logger.info("Listing available Gemini models from Google Generative AI:")
        try:
            for m in genai.list_models():
                logger.info("  name=%s | supported=%s", m.name, m.supported_generation_methods)
        except Exception as ex:
            logger.warning(
                "Failed to list Gemini models. Possibly limited or missing permissions: %s", ex
            )
        return genai

    def _normalize_gemini_model_name(self, raw_name: str) -> str:
        """
        Ensures the model name is recognized by the Gemini API, e.g. 'models/gemini-1.5-flash'.
        If the model is not recognized, fallback to a known valid model.
        """
        if not (raw_name.startswith("models/") or raw_name.startswith("tunedModels/")):
            raw_name = f"models/{raw_name}"

        try:
            available = [m.name for m in genai.list_models()]
            if raw_name not in available:
                logger.warning(
                    "Gemini model '%s' not recognized by the API. Using 'models/gemini-1.5-flash'.",
                    raw_name
                )
                return "models/gemini-1.5-flash"
        except Exception as ex:
            logger.warning(
                "Unable to confirm Gemini model availability. Defaulting to 'models/gemini-1.5-flash'. Error: %s",
                ex
            )
            return "models/gemini-1.5-flash"

        return raw_name

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def forward(self, request: ChatRequest) -> ChatResponse:
        """
        Sends a content-generation request to the specified Gemini model.
        Raises InvalidPromptError if the prompt is missing,
        and ProviderAPIError for any underlying exceptions.
        """
        if not request.prompt:
            raise InvalidPromptError("Gemini prompt cannot be empty.")

        logger.info(
            "Gemini forward_chat() invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.model_name,
                "prompt_length": len(request.prompt),
            },
        )

        final_model_ref = self._normalize_gemini_model_name(self.model_info.model_name)

        # 1) Convert universal ChatRequest -> GeminiChatParameters
        gemini_params = GeminiChatParameters(**request.dict(exclude={"provider_params"}))
        gemini_kwargs = gemini_params.to_gemini_kwargs()

        # 2) Merge additional provider_params
        if request.provider_params:
            gemini_kwargs.update(request.provider_params)

        try:
            model = GenerativeModel(final_model_ref)
            gen_config = types.GenerationConfig(**gemini_kwargs["generation_config"])

            response = model.generate_content(
                request.prompt,
                generation_config=gen_config,
                # provider_params might also contain other recognized fields
                **{k: v for k, v in gemini_kwargs.items() if k != "generation_config"}
            )

            logger.debug("Gemini usage_metadata from response=%r", response.usage_metadata)

            text = response.text
            if not text:
                raise ProviderAPIError("Gemini returned no text.")

            return ChatResponse(
                data=text,
                raw_output=response,
                usage=self.calculate_usage(response)
            )

        except NotFound as nf:
            logger.exception("Gemini model not found or not accessible: %s", nf)
            raise ProviderAPIError(str(nf)) from nf
        except Exception as exc:
            logger.exception("Error in GeminiModel.forward_chat()")
            raise ProviderAPIError(str(exc)) from exc

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        """
        Parses usage_metadata from the GenerateContentResponse to extract usage stats.
        """
        usage_data = getattr(raw_output, "usage_metadata", None)
        if not usage_data:
            logger.debug("No usage_metadata found in raw_output.")
            return UsageStats()

        prompt_count = getattr(usage_data, "prompt_token_count", 0)
        completion_count = getattr(usage_data, "candidates_token_count", 0)
        total_count = getattr(usage_data, "total_token_count", 0) or (prompt_count + completion_count)

        cost_input = (prompt_count / 1000.0) * self.model_info.cost.input_cost_per_thousand
        cost_output = (completion_count / 1000.0) * self.model_info.cost.output_cost_per_thousand
        total_cost = round(cost_input + cost_output, 6)

        return UsageStats(
            total_tokens=total_count,
            prompt_tokens=prompt_count,
            completion_tokens=completion_count,
            cost_usd=total_cost,
        )
