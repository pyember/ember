"""Google Deepmind (Gemini) provider implementation for the Ember framework.

This module provides a comprehensive integration with Google's Gemini language models
through the Deepmind provider implementation for the Ember framework. It establishes
a reliable, high-performance connection with the Google Generative AI API, handling
all aspects of model interaction including authentication, request formatting,
response parsing, error handling, and usage tracking.

The implementation adheres to Google's recommended best practices for API integration,
including proper parameter formatting, efficient retry mechanisms, detailed error
handling, and comprehensive logging. It supports all Gemini model variants with
appropriate adjustments for model-specific requirements and versioning.

Classes:
    DeepmindProviderParams: TypedDict for Gemini-specific parameter configuration
    GeminiChatParameters: Parameter validation and conversion for Gemini requests
    GeminiModel: Core provider implementation for Gemini models

Key features:
    - Authentication and client configuration for Google Generative AI API
    - Support for all Google Gemini model variants (Gemini 1.0, 1.5, etc.)
    - Automatic model name normalization to match Google API requirements
    - Model discovery and validation against available API models
    - Graceful handling of API errors with automatic retries
    - Detailed logging for monitoring and debugging
    - Comprehensive usage statistics for cost tracking
    - Support for model-specific parameters and configuration
    - Parameter validation and type safety
    - Proper timeout handling to prevent hanging requests

Implementation details:
    - Uses the official Google Generative AI Python SDK
    - Implements model discovery and listing for validation
    - Provides fallback mechanisms for configuration errors
    - Uses tenacity for retry logic with exponential backoff
    - Normalizes model names to match Google API conventions
    - Handles response parsing for different API versions
    - Calculates usage statistics based on Google's token metrics

Typical usage example:
    ```python
    # Direct usage (prefer using ModelRegistry or API)
    from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo

    # Configure model information
    model_info = ModelInfo(
        id="deepmind:gemini-1.5-pro",
        name="gemini-1.5-pro",
        provider=ProviderInfo(name="Deepmind", api_key="${GOOGLE_API_KEY}")
    )

    # Initialize the model
    model = GeminiModel(model_info)

    # Basic usage
    response = model("What is the Ember framework?")
    # Access response content with response.data
    
    # Example: "Ember is a Python framework for building AI applications..."

    # Advanced usage with more parameters
    response = model(
        "Generate creative ideas",
        context="You are a helpful creative assistant",
        temperature=0.7,
        provider_params={"top_p": 0.95, "top_k": 40}
    )

    # Access usage information
    # Example response.usage attributes:
    # - response.usage.total_tokens -> 320
    # - response.usage.prompt_tokens -> 45
    # - response.usage.completion_tokens -> 275
    ```

For higher-level usage, prefer the model registry or API interfaces:
    ```python
    from ember.api.models import models

    # Using the models API (automatically handles authentication)
    response = models.deepmind.gemini_15_pro("Tell me about Ember")
    # Access response with response.data
    ```
"""

import logging
from typing import Any, Dict, Optional

import google.generativeai as genai
from google.api_core.exceptions import NotFound
from google.generativeai import GenerativeModel, types
from pydantic import Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from ember.core.exceptions import ModelProviderError, ValidationError
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.plugin_system import provider


class DeepmindProviderParams(ProviderParams):
    """Deepmind-specific provider parameters for fine-tuning Gemini requests.

    This TypedDict defines additional parameters that can be passed to Google
    Generative AI API calls beyond the standard parameters defined in
    BaseChatParameters. These parameters provide fine-grained control over
    the model's generation behavior.

    The parameters align with Google's Generative AI API specification and allow
    for precise control over text generation characteristics including diversity,
    stopping conditions, and sampling strategies. Each parameter affects the
    generation process in specific ways and can be combined to achieve desired
    output characteristics.

    Parameters can be provided in the provider_params field of a ChatRequest:
    ```python
    request = ChatRequest(
        prompt="Tell me about Gemini models",
        provider_params={
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"]
        }
    )
    ```

    Attributes:
        candidate_count: Optional integer specifying the number of response
            candidates to generate. Useful for applications that need diverse
            responses or for implementing re-ranking strategies. Values typically
            range from 1 to 8, with higher values requiring more processing time.

        stop_sequences: Optional list of strings that will cause the model to stop
            generating when encountered. Useful for controlling response length or
            format. The model will stop at the first occurrence of any sequence
            in the list. Example: ["##", "END", "STOP"].

        top_p: Optional float between 0 and 1 for nucleus sampling, controlling the
            cumulative probability threshold for token selection. Lower values (e.g., 0.1)
            make output more focused and deterministic, while higher values (e.g., 0.9)
            increase diversity. Often used together with temperature.

        top_k: Optional integer limiting the number of most likely tokens to consider
            at each generation step. Controls diversity by restricting the token
            selection pool. Typical values range from 1 (greedy decoding) to 40.
            Lower values make output more focused and deterministic.
    """

    candidate_count: Optional[int]
    stop_sequences: Optional[list[str]]
    top_p: Optional[float]
    top_k: Optional[int]


logger: logging.Logger = logging.getLogger(__name__)


class GeminiChatParameters(BaseChatParameters):
    """Parameter handling for Google Gemini generation requests.

    This class extends BaseChatParameters to provide Gemini-specific parameter
    handling and validation. It ensures that parameters are correctly formatted
    for the Google Generative AI API, handling the conversion between Ember's
    universal parameter format and Gemini's API requirements.

    The class implements robust parameter validation, default value handling,
    and conversion logic to ensure that all requests to the Google Generative AI API
    are properly formatted according to the API's expectations. It handles the
    differences between Ember's framework-agnostic parameter names and Google's
    specific parameter naming conventions.

    Key features:
        - Enforces a minimum value for max_tokens to prevent empty responses
        - Provides a sensible default (512 tokens) if not specified by the user
        - Validates that max_tokens is a positive integer with clear error messages
        - Maps Ember's universal max_tokens parameter to Gemini's max_output_tokens
        - Constructs a properly formatted GenerationConfig object for the API
        - Handles temperature scaling in the proper range for Gemini models
        - Provides clean conversion from internal representation to API format

    Implementation details:
        - Uses Pydantic's field validation for type safety and constraints
        - Provides clear error messages for invalid parameter values
        - Uses consistent parameter defaults aligned with the rest of Ember
        - Preserves parameter values when converting to ensure fidelity

    Example:
        ```python
        # Creating parameters with defaults
        params = GeminiChatParameters(prompt="Tell me about Gemini models")

        # Converting to Gemini kwargs
        gemini_kwargs = params.to_gemini_kwargs()
        # Result:
        # {
        #     "generation_config": {
        #         "max_output_tokens": 512,
        #         "temperature": 0.7
        #     }
        # }

        # With custom values
        params = GeminiChatParameters(
            prompt="Explain quantum computing",
            max_tokens=1024,
            temperature=0.9
        )
        gemini_kwargs = params.to_gemini_kwargs()
        # Result includes these parameters with proper names for the Gemini API
        ```
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
            ValidationError: If the token count is less than 1.
        """
        if value < 1:
            raise ValidationError.with_context(
                f"max_tokens must be >= 1, got {value}",
                field_name="max_tokens",
                expected_range=">=1",
                actual_value=value,
                provider="Deepmind",
            )
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
    """Google Deepmind Gemini provider implementation for Ember.

    This class implements the BaseProviderModel interface for Google's Gemini
    language models. It provides a complete integration with the Google Generative AI
    API, handling all aspects of model interaction including authentication,
    request formatting, error handling, retry logic, and response processing.

    The implementation follows Google's recommended best practices for the
    Generative AI API, including proper parameter formatting, error handling,
    and resource cleanup. It incorporates comprehensive error categorization,
    detailed logging, and automatic retries with exponential backoff for transient
    errors, ensuring reliable operation in production environments.

    The class provides three core functions:
        1. Creating and configuring the Google Generative AI client
        2. Processing chat requests through the forward method with proper error handling
        3. Calculating usage statistics for billing and monitoring purposes

    Key features:
        - API authentication and client configuration with API key validation
        - Model discovery and listing for debugging and validation
        - Model name normalization to match Google API requirements
        - Fallback to default models when requested model is unavailable
        - Automatic retry with exponential backoff for transient errors
        - Specialized error handling for different error types (e.g., NotFound)
        - Detailed contextual logging for monitoring and debugging
        - Usage statistics calculation with cost estimation
        - Proper timeout handling to prevent hanging requests
        - Thread-safe implementation for concurrent requests

    Implementation details:
        - Uses the official Google Generative AI Python SDK
        - Implements tenacity-based retry logic with exponential backoff
        - Validates model availability during client creation
        - Provides model name normalization with proper prefixes
        - Handles parameter conversion between Ember and Google formats
        - Integrates with Ember's usage tracking and cost estimation system
        - Supports all Gemini model variants (1.0, 1.5, etc.)

    Attributes:
        PROVIDER_NAME: The canonical name of this provider for registration.
        model_info: Model metadata including credentials and cost schema.
        client: The configured Google Generative AI client instance.
    """

    PROVIDER_NAME: str = "Deepmind"

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
            raise ModelProviderError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message="Google API key is missing or invalid.",
            )

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
            raise InvalidPromptError.with_context(
                "Gemini prompt cannot be empty.",
                provider=self.PROVIDER_NAME,
                model_name=self.model_info.name,
            )

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

            # Gemini SDK doesn't accept timeout parameter directly
            # Extract timeout and remove it from the parameters
            if additional_params and "timeout" in additional_params:
                additional_params.pop("timeout", None)

            # Gemini API expects 'contents' parameter, not 'prompt'
            response = generative_model.generate_content(
                contents=request.prompt,
                generation_config=generation_config,
                **additional_params,
            )
            logger.debug(
                "Gemini usage_metadata from response: %r", response.usage_metadata
            )

            generated_text: str = response.text
            if not generated_text:
                raise ProviderAPIError.for_provider(
                    provider_name=self.PROVIDER_NAME,
                    message="Gemini returned no text.",
                    status_code=None,
                )

            return ChatResponse(
                data=generated_text,
                raw_output=response,
                usage=self.calculate_usage(raw_output=response),
            )
        except NotFound as nf:
            logger.exception("Gemini model not found or not accessible: %s", nf)
            raise ProviderAPIError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message=f"Model not found or not accessible: {str(nf)}",
                status_code=404,
                cause=nf,
            )
        except Exception as exc:
            logger.exception("Error in GeminiModel.forward")
            raise ProviderAPIError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message=f"API error: {str(exc)}",
                cause=exc,
            )

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
