"""Anthropic Claude provider implementation for the Ember framework.

This module provides a comprehensive integration with Anthropic's Claude language models,
implementing the provider interface defined by the Ember framework. It handles
all aspects of communicating with the Anthropic API, including auth, request formatting,
response parsing, error handling, and usage tracking.

The implementation conforms to Anthropic's "best practices" for API integration,
supporting both the legacy Claude prompt format and the modern messages API format.
It handles automatic retries for transient errors, detailed logging, and
error handling to ensure reliability in prod environments.

Key classes:
    AnthropicProviderParams: TypedDict for Anthropic-specific parameters
    AnthropicConfig: Helper for loading and caching model configuration
    AnthropicChatParameters: Parameter conversion for Anthropic chat requests
    AnthropicModel: Core provider implementation for Anthropic models

Details:
    - Authentication and client configuration for Anthropic API
    - Parameter validation and transformation
    - Structured error handling with detailed logging
    - Usage statistics calculation for cost tracking
    - Automatic retries with exponential backoff
    - Thread-safe implementation for concurrent requests
    - Support for both legacy and modern Anthropic API endpoints
    - Uses the official Anthropic Python SDK
    - Handles API versioning and compatibility
    - Provides fallback mechanisms for configuration errors
    - Implements proper timeout handling to prevent hanging requests
    - Calculates token usage for cost estimation and monitoring

Usage example:
    ```python
    # Direct usage (prefer using ModelRegistry or API)
    from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo

    # Configure model information
    model_info = ModelInfo(
        id="anthropic:claude-3-sonnet",
        name="claude-3-sonnet",
        provider=ProviderInfo(name="Anthropic", api_key="sk-ant-...")
    )

    # Initialize the model
    model = AnthropicModel(model_info)

    # Generate a response
    response = model("What is the Ember framework?")
    # Access response content with response.data
    
    # Example: "Ember is a framework for building composable LLM applications..."

    # Access usage statistics
    # Example: response.usage.total_tokens -> 256
    ```

For higher-level usage, prefer the model registry or API interfaces:
    ```python
    from ember.api.models import models

    # Using the models API (automatically handles authentication)
    response = models.anthropic.claude_3_sonnet("Tell me about Ember")
    # Access response with response.data
    ```
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Set

import anthropic
import yaml
from pydantic import Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
    ValidationError,
)
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.plugin_system import provider


class AnthropicProviderParams(ProviderParams):
    """Anthropic-specific provider parameters for fine-tuning API requests.

    This TypedDict defines additional parameters that can be passed to Anthropic API
    calls beyond the standard parameters defined in BaseChatParameters. These parameters
    provide fine-grained control over the model's generation behavior.

    The parameters align with Anthropic's API specification and allow for precise
    control over text generation characteristics including diversity, stopping
    conditions, and sampling strategies. Each parameter affects the generation
    process in specific ways and can be combined to achieve desired output
    characteristics.

    Parameters can be provided in the provider_params field of a ChatRequest:
    ```python
    request = ChatRequest(
        prompt="Tell me about Claude models",
        provider_params={
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"]
        }
    )
    ```

    Attributes:
        top_k: Optional integer limiting the number of most likely tokens to consider
            at each generation step. Controls diversity by restricting the token
            selection pool. Typical values range from 1 (greedy decoding) to 40.
            Lower values make output more focused and deterministic.

        top_p: Optional float between 0 and 1 for nucleus sampling, controlling the
            cumulative probability threshold for token selection. Lower values (e.g., 0.1)
            make output more focused and deterministic, while higher values (e.g., 0.9)
            increase diversity. Often used together with temperature.

        stop_sequences: Optional list of strings that will cause the model to stop
            generating when encountered. Useful for controlling response length or
            format. The model will stop at the first occurrence of any sequence
            in the list. Example: ["Human:", "END", "STOP"].

        stream: Optional boolean to enable streaming responses instead of waiting
            for the complete response. When enabled, tokens are sent as they are
            generated rather than waiting for the complete response. Useful for
            real-time applications and gradual UI updates.
    """

    top_k: Optional[int]
    top_p: Optional[float]
    stop_sequences: Optional[list[str]]
    stream: Optional[bool]


logger: logging.Logger = logging.getLogger(__name__)


class AnthropicConfig:
    """Helper class to load and cache Anthropic configuration from a YAML file.

    This class provides methods to load, cache, and retrieve configuration data for
    Anthropic models. It implements a simple caching mechanism to avoid repeated
    disk reads, improving performance for subsequent accesses.

    The configuration file should contain model registry information, including
    supported model names and their capabilities. If the configuration file cannot
    be found or parsed, the class falls back to default values.

    Implementation details:
    - Uses a class-level cache for efficient access
    - Loads configuration from an anthropic_config.yaml file in the same directory
    - Provides helper methods to retrieve valid models and default model names
    - Handles missing or invalid configuration gracefully with fallback values
    """

    _config_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """Load and cache the Anthropic configuration from a YAML file.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration settings.
        """
        if cls._config_cache is None:
            config_path: str = os.path.join(
                os.path.dirname(__file__), "anthropic_config.yaml"
            )
            try:
                with open(config_path, "r", encoding="utf-8") as config_file:
                    cls._config_cache = yaml.safe_load(config_file)
            except Exception as error:
                logger.warning("Could not load Anthropic config file: %s", error)
                cls._config_cache = {}
        return cls._config_cache

    @classmethod
    def get_valid_models(cls) -> Set[str]:
        """Retrieve valid model names from the configuration.

        Scans the configuration for both full and short form model identifiers.

        Returns:
            Set[str]: A set of valid model names.
        """
        config: Dict[str, Any] = cls.load_config()
        models: List[Dict[str, Any]] = config.get("registry", {}).get("models", [])
        valid_models: Set[str] = set()
        for model in models:
            if "model_name" in model:
                valid_models.add(model["model_name"])
            if "model_id" in model:
                # Also add the short form (assuming 'provider:shortname' format).
                valid_models.add(model["model_id"].split(":")[-1])
        if not valid_models:
            valid_models = {
                "claude-3-5-sonnet",
                "claude-3.7-sonnet",
                "claude-3-opus",
                "claude-3-haiku",
            }
        return valid_models

    @classmethod
    def get_default_model(cls) -> str:
        """Retrieve the default model defined in the configuration.

        Returns:
            str: The default model name.
        """
        config: Dict[str, Any] = cls.load_config()
        models: List[Dict[str, Any]] = config.get("registry", {}).get("models", [])
        if models:
            first_model: Dict[str, Any] = models[0]
            default_model: str = first_model.get("model_name") or first_model.get(
                "model_id", "claude-2"
            )
            return (
                default_model.split(":")[-1] if ":" in default_model else default_model
            )
        return "claude-2"


class AnthropicChatParameters(BaseChatParameters):
    """Parameters for Anthropic chat requests with validation and conversion logic.

    This class extends BaseChatParameters to provide Anthropic-specific parameter
    handling and validation. It ensures that parameters are correctly formatted
    for the Anthropic API, handling the conversion between Ember's universal
    parameter format and Anthropic's API requirements.

    The class implements robust parameter validation, default value handling,
    and conversion logic to ensure that all requests to the Anthropic API are
    properly formatted according to Anthropic's expectations. It handles the
    differences between Ember's framework-agnostic parameter names and Anthropic's
    specific parameter naming conventions.

    Key features:
        - Enforces a minimum value for max_tokens (required by Anthropic API)
        - Provides sensible defaults for required parameters (768 tokens)
        - Validates that max_tokens is a positive integer with clear error messages
        - Converts Ember's universal parameter format to Anthropic's expected format
        - Formats the prompt according to Anthropic's Human/Assistant convention
        - Handles context integration with proper formatting and spacing
        - Supports both legacy prompt format and modern messages API

    Implementation details:
        - Uses Pydantic's field validation for type safety and constraints
        - Provides clear error messages for invalid parameter values
        - Uses consistent parameter defaults aligned with Anthropic recommendations
        - Formats prompts for compatibility with all Claude model versions
        - Maintains backward compatibility with test suite through hybrid approach

    Example:
        ```python
        # Creating parameters with defaults
        params = AnthropicChatParameters(prompt="Tell me about Claude models")

        # Converting to Anthropic kwargs
        anthropic_kwargs = params.to_anthropic_kwargs()
        # Result:
        # {
        #     "prompt": "\\n\\nHuman: Tell me about Claude models\\n\\nAssistant:",
        #     "max_tokens_to_sample": 768,
        #     "temperature": 0.7
        # }

        # With context
        params = AnthropicChatParameters(
            prompt="Explain quantum computing",
            context="You are an expert in physics",
            max_tokens=1024,
            temperature=0.5
        )
        # The context is properly integrated into the prompt format
        ```
    """

    max_tokens: Optional[int] = Field(default=None)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def enforce_default_if_none(cls, value: Optional[int]) -> int:
        """Enforce a default for max_tokens if not provided.

        Args:
            value (Optional[int]): The provided token count, which may be None.

        Returns:
            int: The token count (768 if no value is provided).
        """
        return 768 if value is None else value

    @field_validator("max_tokens")
    @classmethod
    def ensure_positive(cls, value: int) -> int:
        """Ensure that max_tokens is a positive integer.

        Args:
            value (int): The token count to validate.

        Returns:
            int: The validated token count.

        Raises:
            ValidationError: If the token count is less than 1.
        """
        if value < 1:
            raise ValidationError(
                message=f"max_tokens must be >= 1, got {value}",
                context={"max_tokens": value, "min_allowed": 1},
            )
        return value

    def to_anthropic_kwargs(self) -> Dict[str, Any]:
        """Convert chat parameters to keyword arguments for the Anthropic API.

        Prepares the parameters for the messages.create API, but the actual message
        is constructed in the forward method.

        Returns:
            Dict[str, Any]: A dictionary of parameters for the Anthropic messages API.
        """
        # Store the prompt for conversion to message format in the forward method
        # This allows backward compatibility with the test suite
        kwargs: Dict[str, Any] = {
            "prompt": f"{self.context or ''}\n\nHuman: {self.prompt}\n\nAssistant:",
            "max_tokens_to_sample": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs


@provider("Anthropic")
class AnthropicModel(BaseProviderModel):
    """Concrete implementation for interacting with Anthropic models (e.g., Claude).

    This class provides the full implementation of BaseProviderModel for Anthropic's
    Claude language models. It handles client creation, request processing, response
    handling, and usage calculation. The implementation supports all Claude model
    variants including Claude 3, Claude 3.5, and future versions.

    The class provides three core functions:
    1. Creating and configuring the Anthropic API client
    2. Processing chat requests through the forward method
    3. Calculating usage statistics for billing and monitoring

    Key features:
        - Model name normalization against configuration
        - Automatic retry with exponential backoff for transient errors
        - Comprehensive error handling for API failures
        - Support for both legacy prompt format and modern messages API
        - Detailed logging for monitoring and debugging
        - Usage statistics calculation for cost estimation

    Implementation details:
        - Uses the official Anthropic Python SDK
        - Implements tenacity-based retry logic
        - Properly handles API timeouts to prevent hanging
        - Provides cross-version compatibility for API changes
        - Calculates usage statistics based on available data

    Attributes:
        PROVIDER_NAME: The canonical name of this provider for registration.
        model_info: Model metadata including credentials and cost schema.
        client: The configured Anthropic API client instance.
    """

    PROVIDER_NAME: str = "Anthropic"

    def _normalize_anthropic_model_name(self, raw_name: str) -> str:
        """Normalize the provided model name against the configuration.

        Maps model names from the ember format to the exact API model identifier.
        If the supplied model name is unrecognized, the method falls back to a default model.

        Args:
            raw_name (str): The model name provided by the user.

        Returns:
            str: A valid model name for the Anthropic API.
        """
        # Direct mapping from ember model names to Anthropic API model IDs
        model_mapping = {
            # Model ID to raw API name
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
            "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
            # Handle split IDs (provider:model)
            "anthropic:claude-3-opus": "claude-3-opus-20240229",
            "anthropic:claude-3-haiku": "claude-3-haiku-20240307",
            "anthropic:claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
            "anthropic:claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        }

        # If the model is directly in our mapping, use it
        if raw_name in model_mapping:
            return model_mapping[raw_name]

        # Check if it's already a valid dated version
        valid_models: Set[str] = AnthropicConfig.get_valid_models()
        if raw_name in valid_models:
            return raw_name

        # Fallback to default model
        default_model: str = "claude-3-5-sonnet-20240620"  # Most reliable fallback
        logger.warning(
            "Anthropic model '%s' not recognized in configuration. Falling back to '%s'.",
            raw_name,
            default_model,
        )
        return default_model

    def create_client(self) -> anthropic.Anthropic:
        """Instantiate and return an Anthropic API client.

        Retrieves and validates the API key from the model information.

        Returns:
            anthropic.Anthropic: An Anthropic client instance using the provided API key.

        Raises:
            ProviderAPIError: If the API key is missing or invalid.
        """
        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("Anthropic API key is missing or invalid.")
        return anthropic.Anthropic(api_key=api_key)

    @retry(
        wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True
    )
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Forward a chat request to the Anthropic API and process the response.

        Converts a universal ChatRequest into Anthropic-specific parameters,
        invokes the API using keyword arguments, and converts the API response
        into a standardized ChatResponse.

        Args:
            request (ChatRequest): A chat request containing the prompt and provider parameters.

        Returns:
            ChatResponse: A standardized chat response with text, raw output, and usage statistics.

        Raises:
            InvalidPromptError: If the request prompt is empty.
            ProviderAPIError: If an error occurs during API communication.
        """
        if not request.prompt:
            raise InvalidPromptError("Anthropic prompt cannot be empty.")

        correlation_id: str = str(
            request.provider_params.get("correlation_id", uuid.uuid4())
        )
        logger.info(
            "Anthropic forward() invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.name,
                "correlation_id": correlation_id,
                "prompt_length": len(request.prompt),
            },
        )

        final_model_name: str = self._normalize_anthropic_model_name(
            self.model_info.name
        )
        anthropic_params: AnthropicChatParameters = AnthropicChatParameters(
            **request.model_dump(exclude={"provider_params"})
        )
        anthro_kwargs: Dict[str, Any] = anthropic_params.to_anthropic_kwargs()
        anthro_kwargs.update(request.provider_params)

        try:
            # Convert to messages API format
            messages = [{"role": "user", "content": anthro_kwargs.pop("prompt", "")}]

            # Extract timeout from parameters or use default
            timeout = anthro_kwargs.pop("timeout", 30)

            response: Any = self.client.messages.create(
                model=final_model_name,
                messages=messages,
                max_tokens=anthro_kwargs.pop("max_tokens_to_sample", 768),
                timeout=timeout,
                **anthro_kwargs,
            )
            # New API returns content array, join if multiple parts
            if hasattr(response, "content") and isinstance(response.content, list):
                response_text = "".join(
                    [part.text for part in response.content if hasattr(part, "text")]
                )
            else:
                # Fallback for older response format or testing
                response_text = getattr(response, "completion", "").strip()

            usage: UsageStats = self.calculate_usage(raw_output=response)
            return ChatResponse(data=response_text, raw_output=response, usage=usage)
        except Exception as error:
            logger.exception("Anthropic model execution error.")
            raise ProviderAPIError(f"Error calling Anthropic: {error}") from error

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        """Calculate usage statistics based on the API response.

        Uses the usage information from the API response if available,
        otherwise estimates by counting words.

        Args:
            raw_output (Any): The raw response object from the Anthropic API.

        Returns:
            UsageStats: An object containing token counts and cost metrics.
        """
        # Check if there's usage information in the response (new API)
        if hasattr(raw_output, "usage"):
            input_tokens = getattr(raw_output.usage, "input_tokens", 0)
            output_tokens = getattr(raw_output.usage, "output_tokens", 0)
            return UsageStats(
                total_tokens=input_tokens + output_tokens,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cost_usd=0.0,  # Cost calculation would need model-specific rates
            )

        # Fallback for testing or older format responses
        if hasattr(raw_output, "completion"):
            completion_words: int = len(raw_output.completion.split())
            return UsageStats(
                total_tokens=completion_words,
                prompt_tokens=completion_words,
                completion_tokens=0,
                cost_usd=0.0,
            )

        # Content field in newer API responses
        if hasattr(raw_output, "content") and isinstance(raw_output.content, list):
            content_text = "".join(
                [part.text for part in raw_output.content if hasattr(part, "text")]
            )
            content_words = len(content_text.split())
            return UsageStats(
                total_tokens=content_words,
                prompt_tokens=0,
                completion_tokens=content_words,
                cost_usd=0.0,
            )

        # Final fallback
        return UsageStats(
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
        )
