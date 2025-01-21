import logging
from typing import Any

try:
    import ibm_watsonx_ai.foundation_models as ibm_models
except ImportError:
    ibm_models = None  # handle gracefully if not installed

from src.avior.registry.model.provider_registry.base import BaseProviderModel
from src.avior.registry.model.exceptions import InvalidPromptError, ProviderAPIError
from src.avior.registry.model.schemas.chat_schemas import ChatRequest, ChatResponse
from src.avior.registry.model.schemas.usage import UsageStats
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)


class WatsonXModel(BaseProviderModel):
    """
    IBM WatsonX provider implementation.
    """

    PROVIDER_NAME = "WatsonX"

    def create_client(self) -> Any:
        api_key = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("WatsonX API key is missing or invalid.")
        if not ibm_models:
            raise ProviderAPIError("ibm_watsonx_ai not installed.")
        return ibm_models

    @retry(
        wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True
    )
    def forward(self, request: ChatRequest) -> ChatResponse:
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
            response = self.client.Model.generate_text("some_model_id", request.prompt)
            usage = self.calculate_usage(response)
            return ChatResponse(data=response, raw_output=response, usage=usage)
        except Exception as e:
            logger.exception("Error in WatsonXModel.forward()")
            raise ProviderAPIError(str(e)) from e

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        # WatsonX might not return token usage. We'll default to 0.
        return UsageStats()
