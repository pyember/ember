"""OpenAI provider implementation for the Ember framework.

This module provides a comprehensive integration with OpenAI language models
(including GPT-3.5, GPT-4, GPT-4o, and future variants) with the Ember framework.
It handles all aspects of model interaction including auth, request formatting,
response parsing, error handling, retry logic, and usage tracking specifically for
the OpenAI API.

The implementation follows OpenAI's best practices for API integration, including
for parameter handling, retry mechanisms, error categorization,
and logging. It supports all OpenAI model variants with appropriate
parameter adjustments for model-specific requirements.

Classes:
    OpenAIProviderParams: TypedDict defining OpenAI-specific parameters
    OpenAIChatParameters: Parameter conversion for OpenAI chat completions
    OpenAIModel: Core implementation of the OpenAI provider

Details:
    - Authentication and client configuration for OpenAI API
    - Automatic retry with exponential backoff for transient errors
    - Specialized error handling for different error types
    - Model-specific parameter handling (e.g., removing temperature for o1 models)
    - Parameter validation and transformation
    - Detailed logging for monitoring and debugging
    - Usage statistics calculation for cost tracking
    - Proper timeout handling to prevent hanging requests

Usage example:
    ```python
    # Direct usage (prefer using ModelRegistry or API)
    from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo

    # Configure model information
    model_info = ModelInfo(
        id="openai:gpt-4o",
        name="gpt-4o",
        provider=ProviderInfo(name="OpenAI", api_key="sk-...")
    )

    # Initialize the model
    model = OpenAIModel(model_info)

    # Basic usage
    response = model("What is the Ember framework?")
    # Access response content with response.data
    
    # Example: "The Ember framework is a Python library for composable LLM applications..."

    # Advanced usage with more parameters
    response = model(
        "Generate creative ideas",
        context="You are a helpful creative assistant",
        temperature=0.9,
        provider_params={"top_p": 0.95, "frequency_penalty": 0.5}
    )

    # Accessing usage statistics
    # Example: response.usage.total_tokens -> 145
    # Example: response.usage.cost_usd -> 0.000145
    ```

For higher-level usage, prefer the model registry or API interfaces:
    ```python
    from ember.api.models import models

    # Using the models API (automatically handles authentication)
    response = models.openai.gpt4o("Tell me about Ember")
    print(response.data)
    ```
"""

import logging
from typing import Any, Dict, Final, List, Optional, cast

import openai
from pydantic import Field, field_validator
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential

from ember.core.exceptions import ModelProviderError, ValidationError
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.base.utils.usage_calculator import DefaultUsageCalculator
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.plugin_system import provider


class OpenAIProviderParams(ProviderParams):
    """OpenAI-specific provider parameters for fine-tuning API requests.

    This TypedDict defines additional parameters that can be passed to OpenAI API
    calls beyond the standard parameters defined in BaseChatParameters. These parameters
    provide fine-grained control over the model's generation behavior.

    Parameters can be provided in the provider_params field of a ChatRequest:
    ```python
    request = ChatRequest(
        prompt="Generate creative ideas",
        provider_params={
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "stop": ["END"]
        }
    )
    ```

    Attributes:
        stream: Optional boolean to enable streaming responses instead of waiting
            for the complete response.
        stop: Optional list of strings that will cause the model to stop
            generating when encountered.
        presence_penalty: Optional float between -2.0 and 2.0 that penalizes tokens
            based on their presence in the text so far. Positive values discourage
            repetition.
        frequency_penalty: Optional float between -2.0 and 2.0 that penalizes tokens
            based on their frequency in the text so far. Positive values discourage
            repetition.
        top_p: Optional float between 0 and 1 for nucleus sampling, controlling the
            cumulative probability threshold for token selection.
        seed: Optional integer for deterministic sampling, ensuring repeatable outputs
            for the same inputs (when temperature > 0).
    """

    stream: Optional[bool]
    stop: Optional[list[str]]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    top_p: Optional[float]
    seed: Optional[int]


logger: logging.Logger = logging.getLogger(__name__)


class OpenAIChatParameters(BaseChatParameters):
    """Parameters for OpenAI chat requests with validation and conversion logic.

    This class extends BaseChatParameters to provide OpenAI-specific parameter
    handling and validation. It ensures that parameters are correctly formatted
    for the OpenAI API, handling the conversion between Ember's universal
    parameter format and OpenAI's API requirements.

    Key features:
        - Enforces a minimum value for max_tokens
        - Provides a sensible default (512 tokens) if not specified
        - Validates that max_tokens is a positive integer
        - Builds the messages array in the format expected by OpenAI's chat completion API
        - Structures system and user content into proper roles

    The class handles parameter validation and transformation to ensure that
    all requests sent to the OpenAI API are properly formatted and contain
    all required fields with valid values.

    Example:
        ```python
        # With context
        params = OpenAIChatParameters(
            prompt="Tell me about LLMs",
            context="You are a helpful assistant",
            max_tokens=100,
            temperature=0.7
        )
        kwargs = params.to_openai_kwargs()
        # Result:
        # {
        #     "messages": [
        #         {"role": "system", "content": "You are a helpful assistant"},
        #         {"role": "user", "content": "Tell me about LLMs"}
        #     ],
        #     "max_tokens": 100,
        #     "temperature": 0.7
        # }
        ```
    """

    max_tokens: Optional[int] = Field(default=None)

    @field_validator("max_tokens", mode="before")
    def enforce_default_if_none(cls, value: Optional[int]) -> int:
        """Enforce a default value for `max_tokens` if None.

        Args:
            value (Optional[int]): The original max_tokens value, possibly None.

        Returns:
            int: An integer value; defaults to 512 if input is None.
        """
        return 512 if value is None else value

    @field_validator("max_tokens")
    def ensure_positive(cls, value: int) -> int:
        """Validate that `max_tokens` is at least 1.

        Args:
            value (int): The token count provided.

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
                provider="OpenAI",
            )
        return value

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """Convert chat parameters into keyword arguments for the OpenAI API.

        Builds the messages list and returns a dictionary of parameters as expected
        by the OpenAI API.

        Returns:
            Dict[str, Any]: A dictionary containing keys such as 'messages',
            'max_tokens', and 'temperature'.
        """
        messages: List[Dict[str, str]] = []
        if self.context:
            messages.append({"role": "system", "content": self.context})
        messages.append({"role": "user", "content": self.prompt})
        return {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@provider("OpenAI")
class OpenAIModel(BaseProviderModel):
    """Implementation for OpenAI language models in the Ember framework.

    This class provides a comprehensive integration with the OpenAI API, handling
    all aspects of model interaction including authentication, request formatting,
    error handling, retry logic, and response processing. It implements the
    BaseProviderModel interface, making OpenAI models compatible with the wider
    Ember ecosystem.

    The implementation follows OpenAI's best practices for API integration,
    including proper parameter formatting, efficient retry mechanisms, detailed
    error handling, and comprehensive logging. It supports all OpenAI model
    variants with appropriate parameter adjustments for model-specific requirements.

    Key features:
        - Robust error handling with automatic retries for transient errors
        - Specialized handling for different model variants (e.g., o1 models)
        - Comprehensive logging for debugging and monitoring
        - Usage statistics tracking for cost analysis
        - Type-safe parameter handling with runtime validation
        - Model-specific parameter pruning (e.g., removing temperature for o1 models)
        - Proper timeout handling to prevent hanging requests

    The class provides three core functions:
        1. Creating and configuring the OpenAI API client
        2. Processing chat requests through the forward method
        3. Calculating usage statistics for billing and monitoring

    Implementation details:
        - Uses the official OpenAI Python SDK
        - Implements tenacity-based retry logic with exponential backoff
        - Properly handles API timeouts to prevent hanging
        - Calculates usage statistics based on API response data
        - Handles parameter conversion between Ember and OpenAI formats

    Attributes:
        PROVIDER_NAME: The canonical name of this provider for registration.
        model_info: Model metadata including credentials and cost schema.
        client: The configured OpenAI API client instance.
        usage_calculator: Component for calculating token usage and costs.
    """

    PROVIDER_NAME: Final[str] = "OpenAI"

    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize an OpenAIModel instance.

        Args:
            model_info (ModelInfo): Model information including credentials and
                cost schema.
        """
        super().__init__(model_info)
        self.usage_calculator = DefaultUsageCalculator()

    def create_client(self) -> Any:
        """Create and configure the OpenAI client.

        Retrieves the API key from the model information and sets up the OpenAI module.

        Returns:
            Any: The configured OpenAI client module.

        Raises:
            ModelProviderError: If the API key is missing or invalid.
        """
        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ModelProviderError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message="OpenAI API key is missing or invalid.",
            )
        openai.api_key = api_key
        return openai

    def get_api_model_name(self) -> str:
        """Get the model name formatted for OpenAI's API requirements.
        
        OpenAI API requires lowercase model names. This method ensures that
        model names are properly formatted regardless of how they're stored
        internally in the model registry.
        
        Returns:
            str: The properly formatted model name for OpenAI API requests.
        """
        # OpenAI API requires lowercase model names
        return self.model_info.name.lower() if self.model_info.name else ""

    def _prune_unsupported_params(
        self, model_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Remove parameters that are not supported by specific OpenAI models.

        Args:
            model_name (str): The name of the model in use.
            kwargs (Dict[str, Any]): The dictionary of keyword arguments to pass to the API.

        Returns:
            Dict[str, Any]: The pruned dictionary with unsupported keys removed.
        """
        if "o1" in model_name.lower() and "temperature" in kwargs:
            logger.debug("Removing 'temperature' parameter for model: %s", model_name)
            kwargs.pop("temperature")
        return kwargs

    @retry(
        wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True
    )
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Send a ChatRequest to the OpenAI API and process the response.

        Args:
            request (ChatRequest): The chat request containing the prompt along with
                provider-specific parameters.

        Returns:
            ChatResponse: Contains the response text, raw output, and usage statistics.

        Raises:
            InvalidPromptError: If the prompt in the request is empty.
            ProviderAPIError: For any unexpected errors during the API invocation.
        """
        if not request.prompt:
            raise InvalidPromptError.with_context(
                "OpenAI prompt cannot be empty.",
                provider=self.PROVIDER_NAME,
                model_name=self.model_info.name,
            )

        logger.info(
            "OpenAI forward invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.name,
                "prompt_length": len(request.prompt),
            },
        )

        # Convert the universal ChatRequest into OpenAI-specific parameters.
        openai_parameters: OpenAIChatParameters = OpenAIChatParameters(
            **request.model_dump(exclude={"provider_params"})
        )
        openai_kwargs: Dict[str, Any] = openai_parameters.to_openai_kwargs()

        # Merge extra provider parameters in a type-safe manner.
        # Cast the provider_params to OpenAIProviderParams for type safety
        provider_params = cast(OpenAIProviderParams, request.provider_params)
        # Only include non-None values
        openai_kwargs.update(
            {k: v for k, v in provider_params.items() if v is not None}
        )

        # Adjust naming: convert "max_tokens" to "max_completion_tokens" if not already set.
        if (
            "max_tokens" in openai_kwargs
            and "max_completion_tokens" not in openai_kwargs
        ):
            openai_kwargs["max_completion_tokens"] = openai_kwargs.pop("max_tokens")

        # Prune parameters that are unsupported by the current model.
        # Use the normalized model name from our provider-specific method
        openai_kwargs = self._prune_unsupported_params(
            model_name=self.get_api_model_name(),
            kwargs=openai_kwargs,
        )

        try:
            # Use the timeout parameter from the request or the default from BaseChatParameters
            timeout = openai_kwargs.pop("timeout", 30)
            
            # Get properly formatted model name for API using the provider-specific method
            model_name = self.get_api_model_name()
            
            response: Any = self.client.chat.completions.create(
                model=model_name,
                timeout=timeout,
                **openai_kwargs,
            )
            content: str = response.choices[0].message.content.strip()
            usage_stats = self.usage_calculator.calculate(
                raw_output=response,
                model_info=self.model_info,
            )
            return ChatResponse(data=content, raw_output=response, usage=usage_stats)
        except HTTPError as http_err:
            if 500 <= http_err.response.status_code < 600:
                logger.error("OpenAI server error: %s", http_err)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in OpenAIModel.forward()")
            raise ProviderAPIError.for_provider(
                provider_name=self.PROVIDER_NAME,
                message=f"API error: {str(exc)}",
                cause=exc,
            )
