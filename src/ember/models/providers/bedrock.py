"""Amazon Bedrock provider leveraging the Converse API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats


class BedrockConverseProvider(BaseProvider):
    """Provider that proxies chat requests to Amazon Bedrock via Converse."""

    requires_api_key = False

    def __init__(self, api_key: Optional[str] = None, *, region: Optional[str] = None):
        super().__init__(api_key)
        self._boto3 = _import_boto3()

        self._region = region or _config_str("region") or "us-east-1"
        self._client = self._boto3.client("bedrock-runtime", region_name=self._region)

    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        stream_requested = kwargs.pop("stream", False)
        if stream_requested:
            raise ProviderAPIError(
                "Streaming is not implemented for Bedrock Converse API",
                context={"model": model, "error_type": "not_implemented"},
            )

        system = kwargs.pop("system", None)
        context = kwargs.pop("context", None)

        system_prompts = _build_system_prompts(system, context)
        messages = _build_messages(prompt)

        inference_config = _build_inference_config(kwargs)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ProviderAPIError(
                f"Unsupported Bedrock parameters: {unsupported}",
                context={"model": model, "unsupported": tuple(sorted(kwargs))},
            )

        payload: dict[str, Any] = {
            "modelId": model,
            "messages": messages,
        }
        if system_prompts:
            payload["system"] = system_prompts
        if inference_config:
            payload["inferenceConfig"] = inference_config

        try:
            response = self._client.converse(**payload)
        except Exception as exc:
            raise ProviderAPIError(
                f"Bedrock Converse request failed: {exc}",
                context={"model": model, "region": self._region},
            ) from exc

        text = _extract_text(response)
        usage = _usage_from_response(response)
        return ChatResponse(
            data=text,
            usage=usage,
            model_id=model,
            raw_output=response,
        )

    def _get_api_key_from_env(self) -> str:
        return ""


def _import_boto3():
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise ProviderAPIError(
            "boto3 is required for the Bedrock provider. Install with `pip install boto3`.",
            context={"provider": "bedrock"},
        ) from exc
    return boto3


def _config_value(field: str) -> Any:
    from ember._internal.context.runtime import EmberContext

    return EmberContext.current().get_config(f"providers.bedrock.{field}")


def _config_str(field: str) -> Optional[str]:
    raw = _config_value(field)
    if raw is None:
        return None
    if isinstance(raw, str):
        trimmed = raw.strip()
        return trimmed or None
    raise TypeError(f"providers.bedrock.{field} must be a string")


def _build_system_prompts(system: Optional[Any], context: Optional[Any]) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []

    def _append(value: Any) -> None:
        if value:
            prompts.append({"text": str(value)})

    _append(system)
    if context is None:
        return prompts
    if isinstance(context, str):
        _append(context)
    elif isinstance(context, Sequence) and not isinstance(context, (bytes, bytearray, str)):
        for entry in context:
            _append(entry)
    else:
        raise TypeError("Bedrock context must be a string or sequence of strings")

    return prompts


def _build_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"text": prompt},
            ],
        }
    ]


def _build_inference_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}

    if "max_tokens" in kwargs:
        config["maxTokens"] = kwargs.pop("max_tokens")
    if "temperature" in kwargs:
        config["temperature"] = kwargs.pop("temperature")
    if "top_p" in kwargs:
        config["topP"] = kwargs.pop("top_p")
    if "stop" in kwargs:
        stop_sequences = _ensure_string_sequence(kwargs.pop("stop"), "stop")
        config["stopSequences"] = stop_sequences

    return config


def _ensure_string_sequence(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        items = []
        for entry in value:
            if entry is None:
                continue
            items.append(str(entry))
        return items
    raise ProviderAPIError(
        f"{name} must be a string or sequence of strings", context={"parameter": name}
    )


def _extract_text(response: Any) -> str:
    if not response:
        return ""

    output = (
        response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    )
    if not output:
        return ""

    message = (
        output.get("message") if isinstance(output, dict) else getattr(output, "message", None)
    )
    if not message:
        return ""

    content = (
        message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    )
    if not content:
        return ""

    fragments: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if text:
                fragments.append(str(text))
        else:
            text = getattr(block, "text", None)
            if text:
                fragments.append(str(text))
    return "\n".join(fragments)


def _usage_from_response(response: Any) -> Optional[UsageStats]:
    usage = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    )
    if not usage:
        return None

    def _coerce_int(container: Any, key: str) -> int:
        value = container.get(key) if isinstance(container, dict) else getattr(container, key, None)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    prompt_tokens = _coerce_int(usage, "inputTokens")
    completion_tokens = _coerce_int(usage, "outputTokens")
    total_tokens = _coerce_int(usage, "totalTokens")
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
