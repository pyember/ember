import logging
from typing import Any, Dict, Optional

from google.api_core.exceptions import NotFound
import google.generativeai as genai
from google.generativeai import GenerativeModel, types
from pydantic import Field, field_validator
from tenacity import retry, wait_exponential, stop_after_attempt

from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.plugin_system import provider
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ProviderAPIError,
    InvalidPromptError,
)

logger: logging.Logger = logging.getLogger(__name__)


class GeminiChatParameters(BaseChatParameters):
    """Conversion parameters for Google Gemini generation requests.

    This class converts a universal ChatRequest into parameters compliant with the
    Google Gemini API. The `max_tokens` attribute is mapped to the `max_output_tokens`
    field in the GenerationConfig.
    """

    max_tokens: Optional[int] = Field(default=None)

    @field_validator("max_tokens", mode="before")
    def enforce_default_if_none(cls, value: Optional[int]) -> int:
        """Enforce default value for max_tokens.

        Args:
            value (Optional[int]): The supplied token count.

        Returns:
            int: The token count (defaults to 512 if None is provided).
        """
        return 512 if value is None else value

    @field_validator("max_tokens")
    def ensure_positive(cls, value: int) -> int:
        """Ensure that max_tokens is a positive integer.

        Args:
            value (int): The token count.

        Returns:
            int: The validated token count.

        Raises:
            ValueError: If the token count is less than 1.
        """
        if value < 1:
            raise ValueError(f"max_tokens must be >= 1, got {value}")
        return value

    def to_gemini_kwargs(self) -> Dict[str, Any]:
        """Generate keyword arguments for the Gemini API call.

        Returns:
            Dict[str, Any]: A dictionary with the generation configuration and any
            additional parameters.
        """
        generation_config: Dict[str, Any] = {"max_output_tokens": self.max_tokens}
        if self.temperature is not None:  # Only include temperature if provided.
            generation_config["temperature"] = self.temperature
        return {"generation_config": generation_config}


@provider("Deepmind")
class GeminiModel(BaseProviderModel):
    """Deepmind Gemini provider implementation."""

    PROVIDER_NAME: str = "Google"

    def create_client(self) -> Any:
        """Create and configure the Google Generative AI client.

        Configures the google.generativeai SDK using the API key extracted
        from model_info, and logs available Gemini models for debugging.

        Returns:
            Any: The configured google.generativeai client.

        Raises:
            ProviderAPIError: If the API key is missing or invalid.
        """
        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("Google API key is missing or invalid.")

        genai.configure(api_key=api_key)
        logger.info("Listing available Gemini models from Google Generative AI:")
        try:
            for model in genai.list_models():
                logger.info(
                    "  name=%s | supported=%s",
                    model.name,
                    model.supported_generation_methods,
                )
        except Exception as exc:
            logger.warning(
                "Failed to list Gemini models. Possibly limited or missing permissions: %s",
                exc,
            )
        return genai

    def _normalize_gemini_model_name(self, raw_name: str) -> str:
        """Normalize the Gemini model name to the expected API format.

        If `raw_name` does not start with the required prefixes ('models/' or 'tunedModels/'),
        it is prefixed with 'models/'. If the normalized name is not found among the available models,
        a default model name is used.

        Args:
            raw_name (str): The input model name.

        Returns:
            str: A normalized and validated model name.
        """
        if not (raw_name.startswith("models/") or raw_name.startswith("tunedModels/")):
            raw_name = f"models/{raw_name}"

        try:
            available_models = [m.name for m in genai.list_models()]
            if raw_name not in available_models:
                logger.warning(
                    "Gemini model '%s' not recognized by the API. Using 'models/gemini-1.5-flash'.",
                    raw_name,
                )
                return "models/gemini-1.5-flash"
        except Exception as exc:
            logger.warning(
                "Unable to confirm Gemini model availability. Defaulting to 'models/gemini-1.5-flash'. Error: %s",
                exc,
            )
            return "models/gemini-1.5-flash"

        return raw_name

    @retry(
        wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True
    )
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Forward a chat request to the Gemini content generation API.

        Converts a universal ChatRequest to Gemini-specific parameters, sends the
        generation request, and returns a ChatResponse with the generated content and usage stats.

        Args:
            request (ChatRequest): The chat request containing the prompt and additional parameters.

        Returns:
            ChatResponse: The response with generated text and usage statistics.

        Raises:
            InvalidPromptError: If the chat prompt is empty.
            ProviderAPIError: If the provider returns an error or no content.
        """
        if not request.prompt:
            raise InvalidPromptError("Gemini prompt cannot be empty.")

        logger.info(
            "Gemini forward invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.name,
                "prompt_length": len(request.prompt),
            },
        )

        final_model_ref: str = self._normalize_gemini_model_name(self.model_info.name)

        # Convert the universal ChatRequest into Gemini-specific parameters.
        gemini_params: GeminiChatParameters = GeminiChatParameters(
            **request.model_dump(exclude={"provider_params"})
        )
        gemini_kwargs: Dict[str, Any] = gemini_params.to_gemini_kwargs()

        # Merge additional provider parameters if present.
        if request.provider_params:
            gemini_kwargs.update(request.provider_params)

        try:
            generative_model: GenerativeModel = GenerativeModel(final_model_ref)
            generation_config: types.GenerationConfig = types.GenerationConfig(
                **gemini_kwargs["generation_config"]
            )
            additional_params: Dict[str, Any] = {
                key: value
                for key, value in gemini_kwargs.items()
                if key != "generation_config"
            }

            response = generative_model.generate_content(
                prompt=request.prompt,
                generation_config=generation_config,
                **additional_params,
            )
            logger.debug(
                "Gemini usage_metadata from response: %r", response.usage_metadata
            )

            generated_text: str = response.text
            if not generated_text:
                raise ProviderAPIError("Gemini returned no text.")

            return ChatResponse(
                data=generated_text,
                raw_output=response,
                usage=self.calculate_usage(raw_output=response),
            )
        except NotFound as nf:
            logger.exception("Gemini model not found or not accessible: %s", nf)
            raise ProviderAPIError(str(nf)) from nf
        except Exception as exc:
            logger.exception("Error in GeminiModel.forward")
            raise ProviderAPIError(str(exc)) from exc

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        """Calculate usage statistics from the Gemini API response.

        Parses the usage metadata contained in the raw API response to compute token counts
        and cost estimations.

        Args:
            raw_output (Any): The raw response from the Gemini API.

        Returns:
            UsageStats: An object containing the total tokens used, prompt tokens,
            completion tokens, and the calculated cost (in USD).
        """
        usage_data = getattr(raw_output, "usage_metadata", None)
        if not usage_data:
            logger.debug("No usage_metadata found in raw_output.")
            return UsageStats()

        prompt_count: int = getattr(usage_data, "prompt_token_count", 0)
        completion_count: int = getattr(usage_data, "candidates_token_count", 0)
        total_tokens: int = getattr(usage_data, "total_token_count", 0) or (
            prompt_count + completion_count
        )

        input_cost: float = (
            prompt_count / 1000.0
        ) * self.model_info.cost.input_cost_per_thousand
        output_cost: float = (
            completion_count / 1000.0
        ) * self.model_info.cost.output_cost_per_thousand
        total_cost: float = round(input_cost + output_cost, 6)

        return UsageStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_count,
            completion_tokens=completion_count,
            cost_usd=total_cost,
        )
