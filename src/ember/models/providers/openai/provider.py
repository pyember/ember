from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from typing import Any, Final

from tenacity import retry, stop_after_attempt, wait_exponential

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.base import BaseProvider
from ember.models.schemas import (
    ChatResponse,
    EmbeddingResponse,
    UsageStats,
    normalize_reasoning_config,
    normalize_text_config,
)

from ._chat import ChatCompletionsHandler
from ._client import create_openai_client
from ._messages import build_messages, normalize_context
from ._responses import ResponsesHandler

_RESPONSES_PREFIXES: Final[tuple[str, ...]] = ("gpt-5", "gpt-4.1", "o1", "o3", "o4")
_RESPONSES_TRIGGER_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "max_output_tokens",
        "modalities",
        "metadata",
        "response_format",
        "reasoning_budget",
        "parallel_tool_calls",
    }
)
_SUPPORTED_MODELS: Final[set[str]] = {
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-realtime-preview",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-codex",
    "gpt-5-chat-latest",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "o4-mini",
}


class OpenAIProvider(BaseProvider):
    supports_responses_api = True

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self._client = create_openai_client(self.api_key)
        self._chat = ChatCompletionsHandler(self._client)
        self._responses = ResponsesHandler(self._client)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        if kwargs.pop("stream", False):
            raise ProviderAPIError(
                "Streaming is not implemented; use stream_complete() instead",
                context={"model": model},
            )

        system_prompt = kwargs.pop("system", None)
        try:
            context = normalize_context(kwargs.pop("context", None))
        except ProviderAPIError as exc:
            raise ProviderAPIError(str(exc), context={"model": model}) from exc

        messages = build_messages(
            prompt,
            str(system_prompt) if system_prompt is not None else None,
            context,
        )

        try:
            reasoning_cfg = normalize_reasoning_config(kwargs.pop("reasoning", None))
        except (TypeError, ValueError) as exc:
            raise ProviderAPIError(
                f"Invalid reasoning configuration: {exc}",
                context={"model": model, "parameter": "reasoning"},
            ) from exc
        try:
            text_cfg = normalize_text_config(kwargs.pop("text", None))
        except (TypeError, ValueError) as exc:
            raise ProviderAPIError(
                f"Invalid text configuration: {exc}",
                context={"model": model, "parameter": "text"},
            ) from exc

        options = dict(kwargs)
        use_responses = _should_use_responses_api(model, options)
        if use_responses:
            reasoning_payload = reasoning_cfg.to_openai_payload() if reasoning_cfg else None
            text_payload = text_cfg.to_openai_payload() if text_cfg else None
            return self._responses.complete(
                model,
                messages,
                reasoning_payload,
                text_payload,
                options,
            )

        if reasoning_cfg or text_cfg:
            raise ProviderAPIError(
                "reasoning/text controls are only supported on GPT-5 and o-series models",
                context={"model": model},
            )

        return self._chat.complete(model, messages, options)

    def complete_responses_payload(self, payload: dict[str, object], **_: Any) -> ChatResponse:
        return self._responses.complete_from_payload(payload)

    def stream_complete(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> Generator[str, None, ChatResponse]:
        system_prompt = kwargs.pop("system", None)
        try:
            context = normalize_context(kwargs.pop("context", None))
        except ProviderAPIError as exc:
            raise ProviderAPIError(str(exc), context={"model": model}) from exc

        messages = build_messages(
            prompt,
            str(system_prompt) if system_prompt is not None else None,
            context,
        )

        try:
            reasoning_cfg = normalize_reasoning_config(kwargs.pop("reasoning", None))
        except (TypeError, ValueError) as exc:
            raise ProviderAPIError(
                f"Invalid reasoning configuration: {exc}",
                context={"model": model, "parameter": "reasoning"},
            ) from exc
        try:
            text_cfg = normalize_text_config(kwargs.pop("text", None))
        except (TypeError, ValueError) as exc:
            raise ProviderAPIError(
                f"Invalid text configuration: {exc}",
                context={"model": model, "parameter": "text"},
            ) from exc

        options = dict(kwargs)
        use_responses = _should_use_responses_api(model, options)
        if use_responses:
            reasoning_payload = reasoning_cfg.to_openai_payload() if reasoning_cfg else None
            text_payload = text_cfg.to_openai_payload() if text_cfg else None
            return self._responses.stream(
                model,
                messages,
                reasoning_payload,
                text_payload,
                options,
            )

        if reasoning_cfg or text_cfg:
            raise ProviderAPIError(
                "reasoning/text controls are only supported on GPT-5 and o-series models",
                context={"model": model},
            )

        return self._chat.stream(model, messages, options)

    def stream_responses_payload(
        self, payload: dict[str, object], **_: Any
    ) -> Generator[str, None, ChatResponse]:
        return self._responses.stream_from_payload(payload)

    def validate_model(self, model: str) -> bool:
        return model in _SUPPORTED_MODELS or model.startswith("gpt-") or model.startswith("o")

    def get_model_info(self, model: str) -> dict[str, object]:
        info = super().get_model_info(model)
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4o-realtime-preview": 128000,
            "gpt-4.1": 128000,
            "gpt-4.1-mini": 128000,
            "gpt-4.1-preview": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-5": 200000,
            "gpt-5-mini": 128000,
            "gpt-5-nano": 64000,
            "gpt-5-codex": 200000,
            "gpt-5-chat-latest": 200000,
            "o1": 200000,
            "o1-mini": 128000,
            "o1-preview": 200000,
        }
        info.update(
            {
                "context_window": context_windows.get(model, 4096),
                "supports_functions": model.startswith("gpt-") or model.startswith("o"),
                "supports_vision": model
                in {
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4o-realtime-preview",
                    "gpt-4.1",
                    "gpt-4.1-mini",
                },
            }
        )
        return info

    def _get_api_key_from_env(self) -> str:
        from ember._internal.context.runtime import EmberContext

        return EmberContext.current().get_credential("openai")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def embed(
        self,
        inputs: Sequence[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings using OpenAI embedding models."""
        dimensions = kwargs.pop("dimensions", None)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": list(inputs),
        }
        if dimensions is not None:
            if isinstance(dimensions, bool):
                raise TypeError("dimensions must be an integer")
            request_kwargs["dimensions"] = int(dimensions)

        import openai

        try:
            response = self._client.embeddings.create(**request_kwargs)
        except openai.AuthenticationError as exc:
            raise ProviderAPIError(
                "Invalid OpenAI API key",
                context={"model": model},
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderAPIError(
                "OpenAI rate limit exceeded",
                context={"model": model},
            ) from exc
        except openai.APIError as exc:
            raise ProviderAPIError(
                f"OpenAI API error: {exc}",
                context={"model": model},
            ) from exc

        embeddings = [item.embedding for item in response.data]
        usage_payload = getattr(response, "usage", None)
        usage: UsageStats | None = None
        if usage_payload is not None:
            prompt_tokens = getattr(usage_payload, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage_payload, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_payload, "total_tokens", 0) or 0
            usage = UsageStats(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
            )

        resolved_model = getattr(response, "model", None) or model

        return EmbeddingResponse(
            embeddings=embeddings,
            usage=usage,
            model_id=str(resolved_model),
            raw_output=response.model_dump() if hasattr(response, "model_dump") else None,
        )


def _uses_responses_api(model: str) -> bool:
    lowered = model.lower()
    return lowered == "gpt-5" or lowered.startswith(_RESPONSES_PREFIXES)


def _should_use_responses_api(model: str, options: Mapping[str, object]) -> bool:
    if _uses_responses_api(model):
        return True
    for key in options:
        if key in _RESPONSES_TRIGGER_OPTIONS:
            return True
    return False
