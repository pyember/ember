import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Set

import anthropic
import yaml
from pydantic import Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from ember.plugin_system import provider
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.usage import UsageStats

logger: logging.Logger = logging.getLogger(__name__)


class AnthropicConfig:
    """Helper class to load and cache Anthropic configuration from a YAML file.

    This class provides methods to load, cache, and retrieve configuration data for Anthropic models.
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
            valid_models = {"claude-3-", "claude-3.5-sonnet"}
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
    """Parameters for Anthropic chat requests.

    This class converts a universal ChatRequest into parameters compliant with the Anthropic API.
    The API requires a positive integer for 'max_tokens_to_sample', defaulting to 768.
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
            ValueError: If the token count is less than 1.
        """
        if value < 1:
            raise ValueError(f"max_tokens must be >= 1, got {value}")
        return value

    def to_anthropic_kwargs(self) -> Dict[str, Any]:
        """Convert chat parameters to keyword arguments for the Anthropic API.

        Constructs a prompt in Anthropic's required format and maps other parameters accordingly.

        Returns:
            Dict[str, Any]: A dictionary of parameters for the Anthropic completions API.
        """
        anthropic_prompt: str = (
            f"{self.context or ''}\n\nHuman: {self.prompt}\n\nAssistant:"
        )
        kwargs: Dict[str, Any] = {
            "prompt": anthropic_prompt,
            "max_tokens_to_sample": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs


@provider("Anthropic")
class AnthropicModel(BaseProviderModel):
    """Concrete implementation for interacting with Anthropic models (e.g., Claude).

    This class provides methods to create an Anthropic client, forward chat requests, and compute usage statistics.
    """

    PROVIDER_NAME: str = "Anthropic"

    def _normalize_anthropic_model_name(self, raw_name: str) -> str:
        """Normalize the provided model name against the configuration.

        If the supplied model name is unrecognized, the method falls back to a default model.

        Args:
            raw_name (str): The model name provided by the user.

        Returns:
            str: A valid model name from the configuration.
        """
        valid_models: Set[str] = AnthropicConfig.get_valid_models()
        if raw_name not in valid_models:
            default_model: str = AnthropicConfig.get_default_model()
            logger.warning(
                "Anthropic model '%s' not recognized in configuration. Falling back to '%s'.",
                raw_name,
                default_model,
            )
            return default_model
        return raw_name

    def create_client(self) -> anthropic.Client:
        """Instantiate and return an Anthropic API client.

        Retrieves and validates the API key from the model information.

        Returns:
            anthropic.Client: An Anthropic client instance using the provided API key.

        Raises:
            ProviderAPIError: If the API key is missing or invalid.
        """
        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("Anthropic API key is missing or invalid.")
        return anthropic.Client(api_key=api_key)

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

        correlation_id: str = request.provider_params.get(
            "correlation_id", str(uuid.uuid4())
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
            response: Any = self.client.completions.create(
                model=final_model_name, **anthro_kwargs
            )
            response_text: str = response.completion.strip()
            usage: UsageStats = self.calculate_usage(raw_output=response)
            return ChatResponse(data=response_text, raw_output=response, usage=usage)
        except Exception as error:
            logger.exception("Anthropic model execution error.")
            raise ProviderAPIError(f"Error calling Anthropic: {error}") from error

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        """Calculate usage statistics based on the API response.

        Computes usage by counting the words in the API response's completion text.

        Args:
            raw_output (Any): The raw response object from the Anthropic API.

        Returns:
            UsageStats: An object containing token counts and cost metrics.
        """
        completion_words: int = len(raw_output.completion.split())
        return UsageStats(
            total_tokens=completion_words,
            prompt_tokens=completion_words,
            completion_tokens=0,
            cost_usd=0.0,
        )
