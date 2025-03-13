import logging
from typing import Any, Dict, Final, List, Optional, cast

import openai
from pydantic import BaseModel, Field, field_validator
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential

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
    OpenAIProviderParams,
)
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.plugin_system import provider
from ember.core.registry.model.base.utils.usage_calculator import (
    DefaultUsageCalculator,
)

logger: logging.Logger = logging.getLogger(__name__)


class OpenAIChatParameters(BaseChatParameters):
    """Parameters for OpenAI chat requests.

    Ensures that `max_tokens` is not None (defaulting to 512) and assembles the
    OpenAI-style messages list.
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
            ValueError: If the token count is less than 1.
        """
        if value < 1:
            raise ValueError(f"max_tokens must be >= 1, got {value}")
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


# NOTE: This class is deprecated and replaced by OpenAIProviderParams TypedDict
# class OpenAIExtraParams(BaseModel):
#     """Extra provider parameters for OpenAI that may be safely overridden by users."""
#
#     stream: Optional[bool] = None
#     stop: Optional[List[str]] = None
#     # Additional overrideable parameters can be defined here as needed.


@provider("OpenAI")
class OpenAIModel(BaseProviderModel):
    """Implementation for OpenAI-based models.

    Integrates with the OpenAI API, including transient error retry logic,
    specialized error handling, and extraction of usage as well as cost details.
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
            ProviderAPIError: If the API key is missing or invalid.
        """
        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("OpenAI API key is missing or invalid.")
        openai.api_key = api_key
        return openai

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
            raise InvalidPromptError("OpenAI prompt cannot be empty.")

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
        openai_kwargs = self._prune_unsupported_params(
            model_name=self.model_info.name,
            kwargs=openai_kwargs,
        )

        try:
            response: Any = self.client.chat.completions.create(
                model=self.model_info.name,
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
            raise ProviderAPIError(str(exc)) from exc
