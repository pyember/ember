import logging
import openai
from typing import Any
from tenacity import retry, wait_exponential, stop_after_attempt
from requests.exceptions import HTTPError
from pydantic import Field, field_validator, BaseModel

from src.avior.registry.model.provider_registry.base import BaseProviderModel
from src.avior.registry.model.exceptions import InvalidPromptError, ProviderAPIError
from src.avior.registry.model.schemas.usage import UsageStats
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    BaseChatParameters
)

logger = logging.getLogger(__name__)

class OpenAIChatParameters(BaseChatParameters):
    """
    Provider-specific param class for OpenAI. 
    Ensures max_tokens is never None (default=512).
    Builds the OpenAI-style 'messages' list.
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

    def to_openai_kwargs(self) -> dict:
        messages = []
        if self.context:
            messages.append({"role": "system", "content": self.context})
        messages.append({"role": "user", "content": self.prompt})

        kwargs = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # {{ Convert max_tokens -> max_completion_tokens if model demands it }}
        # For example, if the model name starts with "o1", or you detect a special flag:
        # (Adjust this logic to match your actual requirement.)
        return kwargs

class OpenAIExtraParams(BaseModel):
    # Only include parameters that users can override safely
    stream: bool | None = None
    stop: list[str] | None = None
    # etc...

class OpenAIModel(BaseProviderModel):
    """
    Concrete implementation for OpenAI-based models (e.g. GPT-4, GPT-4o).
    Includes:
      - Retry logic for transient errors
      - Specialized error handling
      - Usage cost extraction
    """

    PROVIDER_NAME = "OpenAI"

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)

    def create_client(self) -> Any:
        api_key = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("OpenAI API key is missing or invalid.")
        openai.api_key = api_key
        return openai

    def _prune_unsupported_params(self, model_name: str, kwargs: dict) -> dict:
        """
        Certain OpenAI models (like 'o1') do not allow 'temperature' or 'max_tokens'.
        We remove unsupported keys from kwargs in a single place, keeping code clean.
        """
        # Example: if model_name is "o1" or contains "o1", remove "temperature"
        # (Adjust if you detect 'unsupported_parameter' for other fields.)
        if "o1" in model_name.lower():
            if "temperature" in kwargs:
                logger.debug("Removing 'temperature' param for model='%s'", model_name)
                kwargs.pop("temperature")
        return kwargs

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def forward(self, request: ChatRequest) -> ChatResponse:
        if not request.prompt:
            raise InvalidPromptError("OpenAI prompt cannot be empty.")

        logger.info(
            "OpenAI forward() invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.model_name,
                "prompt_length": len(request.prompt),
            },
        )

        # 1) Convert universal ChatRequest
        openai_params = OpenAIChatParameters(**request.dict(exclude={"provider_params"}))

        # 2) Base OpenAI kwargs from param class (includes "messages")
        openai_kwargs = openai_params.to_openai_kwargs()

        # 3) Parse extra provider params in typed form, merge carefully
        extra_params = OpenAIExtraParams(**request.provider_params)
        typed_overrides = extra_params.model_dump(exclude_unset=True)
        # For example, if user sets "stop" in provider_params, it overrides what's in openai_kwargs
        for k, v in typed_overrides.items():
            if v is not None:
                openai_kwargs[k] = v

        # 4) Adjust naming, prune unsupported
        if "max_tokens" in openai_kwargs and "max_completion_tokens" not in openai_kwargs:
            openai_kwargs["max_completion_tokens"] = openai_kwargs.pop("max_tokens")
        openai_kwargs = self._prune_unsupported_params(
            model_name=self.model_info.model_name, kwargs=openai_kwargs
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_info.model_name,
                **openai_kwargs
            )
            text = response.choices[0].message.content.strip()
            usage = self.calculate_usage(response)
            return ChatResponse(data=text, raw_output=response, usage=usage)

        except HTTPError as e:
            if 500 <= e.response.status_code < 600:
                logger.error("OpenAI server error: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error in OpenAIModel.forward()")
            raise ProviderAPIError(str(e)) from e

    def calculate_usage(self, raw_output: Any) -> UsageStats:
        usage_obj = raw_output.usage
        total = usage_obj.total_tokens
        prompt_toks = usage_obj.prompt_tokens
        completion_toks = usage_obj.completion_tokens

        cost_input = (
            prompt_toks / 1000.0
        ) * self.model_info.cost.input_cost_per_thousand
        cost_output = (
            completion_toks / 1000.0
        ) * self.model_info.cost.output_cost_per_thousand
        total_cost = round(cost_input + cost_output, 6)

        return UsageStats(
            total_tokens=total,
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
            cost_usd=total_cost,
        )
