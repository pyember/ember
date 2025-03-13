import logging
from typing import Any

try:
    import ibm_watsonx_ai.foundation_models as ibm_models
except ImportError:
    ibm_models = None  # Gracefully handle the absence of the ibm_watsonx_ai package.

from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.usage import UsageStats
from tenacity import retry, wait_exponential, stop_after_attempt
from ember.plugin_system import provider

logger: logging.Logger = logging.getLogger(__name__)


@provider("WatsonX")
class WatsonXModel(BaseProviderModel):
    """IBM WatsonX provider implementation."""

    PROVIDER_NAME: str = "WatsonX"

    def create_client(self) -> Any:
        """Create and return a WatsonX client instance.

        Retrieves the API key from the model configuration and verifies the ibm_watsonx_ai
        library is installed. This client will be used for making API calls.

        Raises:
            ProviderAPIError: If the API key is missing or if the ibm_watsonx_ai package is not installed.

        Returns:
            Any: The WatsonX client module.
        """
        api_key: str = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("WatsonX API key is missing or invalid.")
        if ibm_models is None:
            raise ProviderAPIError("ibm_watsonx_ai is not installed.")
        return ibm_models

    @retry(
        wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True
    )
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Forward the chat request to the WatsonX API and return the response.

        Validates the prompt in the request, logs key request details, and invokes the WatsonX
        API using named argument invocation for clarity. In case of any exceptions, wraps
        the error in a ProviderAPIError.

        Args:
            request (ChatRequest): The chat request containing the prompt and other parameters.

        Raises:
            InvalidPromptError: If the provided prompt is empty.
            ProviderAPIError: If the WatsonX API call fails.

        Returns:
            ChatResponse: The response from WatsonX encapsulated in a ChatResponse object.
        """
        if not request.prompt:
            raise InvalidPromptError("WatsonX prompt cannot be empty.")

        logger.info(
            "WatsonX forward() invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.model_name,
                "prompt_length": len(request.prompt),
            },
        )

        try:
            model_id: str = "some_model_id"
            response: Any = self.client.Model.generate_text(
                model_id=model_id,
                prompt=request.prompt,
            )
            usage: UsageStats = self.calculate_usage(raw_output=response)
            return ChatResponse(data=response, raw_output=response, usage=usage)
        except Exception as error:
            logger.exception("Error in WatsonXModel.forward()")
            raise ProviderAPIError(f"Error during WatsonX forward: {error}") from error

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        """Calculate usage statistics from the raw API response.

        As WatsonX might not provide token usage details, this method returns a default
        UsageStats object with zero values.

        Args:
            raw_output (Any): The raw response from the WatsonX API.

        Returns:
            UsageStats: An instance with default usage statistics.
        """
        return UsageStats()
