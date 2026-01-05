"""Azure OpenAI provider built atop the OpenAI Python SDK."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Generator, Optional, cast

import openai

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats

_DEFAULT_API_VERSION = "2024-10-21"

_RESPONSES_ALLOWED_PARAMS = frozenset(
    {
        "max_output_tokens",
        "stop",
        "metadata",
        "response_format",
        "tool_choice",
        "tools",
        "modalities",
        "user",
        "seed",
        "reasoning",
        "input_image_parameters",
    }
)


class AzureOpenAIProvider(BaseProvider):
    """Provider for Azure OpenAI deployments using the OpenAI SDK."""

    supports_responses_api = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        responses_deployments: Optional[Sequence[str]] = None,
        default_use_responses: Optional[bool] = None,
    ):
        """Initialize the Azure OpenAI provider with configurable overrides.

        Args:
            api_key: API key used for authentication. If omitted, the provider
                attempts to read it from the Ember context.
            endpoint: Azure OpenAI endpoint URL (overrides context configuration).
            api_version: API version to use. Defaults to the latest GA version.
            responses_deployments: Iterable of deployment names that require the
                Responses API (e.g. GPT-5 deployments).
            default_use_responses: Fallback flag when deployment name is not in
                `responses_deployments`.

        Raises:
            ValueError: If no endpoint is configured.
        """
        super().__init__(api_key)

        configured_endpoint = endpoint or _config_str("endpoint")
        if not configured_endpoint:
            raise ValueError("providers.azure_openai.endpoint not set in Ember configuration")

        configured_version = api_version or _config_str("api_version") or _DEFAULT_API_VERSION

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=configured_endpoint.rstrip("/") + "/openai/deployments",
            default_query={"api-version": configured_version},
        )
        configured_deployments = responses_deployments
        if configured_deployments is None:
            configured_deployments = _config_list("responses_deployments")
        self._responses_deployments = {
            deployment.lower(): True for deployment in configured_deployments
        }
        if default_use_responses is not None:
            self._default_use_responses = default_use_responses
        else:
            self._default_use_responses = _config_bool("use_responses_api", default=False)

    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        """Generate a completion using an Azure OpenAI deployment."""
        use_responses = kwargs.pop("use_responses_api", None)
        if use_responses is None:
            use_responses = self._should_use_responses(model)

        system = kwargs.pop("system", None)
        context = kwargs.pop("context", None)
        messages = _build_messages(prompt, system, context)

        try:
            if use_responses:
                return self._complete_with_responses(model, messages, **kwargs)
            return self._complete_with_chat(model, messages, **kwargs)
        except ProviderAPIError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ProviderAPIError(
                f"Azure OpenAI error: {exc}",
                context={"model": model},
            ) from exc

    def _complete_with_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        params: Dict[str, Any] = {"model": model, "messages": messages}

        allowed = {"temperature", "max_tokens", "top_p", "stop", "stream"}
        for field in allowed:
            if field in kwargs:
                params[field] = kwargs.pop(field)

        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ProviderAPIError(
                f"Unsupported Azure chat parameters: {unsupported}",
                context={"model": model, "unsupported": tuple(sorted(kwargs))},
            )

        resp = cast(Any, self.client.chat.completions).create(**params)

        text = ""
        if resp.choices:
            text = getattr(resp.choices[0].message, "content", "") or ""

        usage = _usage_from_payload(getattr(resp, "usage", None))
        return ChatResponse(data=text, usage=usage, model_id=model, raw_output=resp)

    def _complete_with_responses(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        stream_requested = kwargs.pop("stream", False)
        if stream_requested:
            raise ProviderAPIError(
                "Streaming is not implemented for Azure Responses API",
                context={"model": model, "error_type": "not_implemented"},
            )

        params: Dict[str, Any] = {
            "model": model,
            "input": _as_responses_turns(messages),
        }

        if "max_tokens" in kwargs:
            params["max_output_tokens"] = kwargs.pop("max_tokens")

        for key in list(kwargs.keys()):
            if key in _RESPONSES_ALLOWED_PARAMS:
                params[key] = kwargs.pop(key)
                continue
            raise ProviderAPIError(
                f"Unsupported parameter for Azure Responses API: {key}",
                context={"model": model, "parameter": key},
            )

        resp = self.client.responses.create(**params)
        text = _extract_text_from_responses(resp)
        usage = _usage_from_payload(getattr(resp, "usage", None))
        return ChatResponse(data=text, usage=usage, model_id=model, raw_output=resp)

    def complete_responses_payload(self, payload: Mapping[str, Any], **kwargs: Any) -> ChatResponse:
        body = _coerce_payload(payload)
        model = body["model"]
        responses_client = getattr(self.client, "responses", None)
        if responses_client is None:
            raise ProviderAPIError(
                "Azure OpenAI SDK does not expose a responses client",
                context={"model": model},
            )
        if body.get("stream"):
            raise ValueError("Use stream_responses_payload for streaming requests")
        try:
            resp = responses_client.create(**body)
        except openai.APIError as exc:
            raise ProviderAPIError(
                f"Azure Responses error: {exc}",
                context={"model": model},
            ) from exc
        text = _extract_text_from_responses(resp)
        usage = _usage_from_payload(getattr(resp, "usage", None))
        response_id = getattr(resp, "id", None)
        return ChatResponse(
            data=text,
            output=_responses_output(resp),
            response_id=str(response_id) if response_id else None,
            usage=usage,
            model_id=model,
            raw_output=_serialize_openai_object(resp),
        )

    def stream_responses_payload(
        self,
        payload: Mapping[str, Any],
        **kwargs: Any,
    ) -> Generator[str, None, ChatResponse]:
        body = _coerce_payload(payload)
        model = body["model"]
        body.pop("stream", None)
        try:
            stream_manager = self.client.responses.stream(**body)  # type: ignore[call-arg]
        except openai.APIError as exc:
            raise ProviderAPIError(
                f"Azure Responses streaming error: {exc}", context={"model": model}
            ) from exc

        def _generator() -> Generator[str, None, ChatResponse]:
            try:
                with stream_manager as stream:
                    for event in stream:
                        payload_obj = _serialize_openai_object(event)
                        yield json.dumps(payload_obj, ensure_ascii=False)
                final_response = stream_manager.get_final_response()
            except openai.APIError as exc:  # pragma: no cover
                raise ProviderAPIError(
                    f"Azure Responses streaming error: {exc}", context={"model": model}
                ) from exc

            text = _extract_text_from_responses(final_response)
            output_items = _responses_output(final_response)
            usage = _usage_from_payload(getattr(final_response, "usage", None))
            response_id = getattr(final_response, "id", None)
            return ChatResponse(
                data=text,
                output=output_items,
                response_id=str(response_id) if response_id else None,
                usage=usage,
                model_id=model,
                raw_output=_serialize_openai_object(final_response),
            )

        return _generator()

    def _should_use_responses(self, deployment: str) -> bool:
        return self._responses_deployments.get(deployment.lower(), self._default_use_responses)

    def _get_api_key_from_env(self) -> str:
        """Return the configured Azure OpenAI API key."""
        from ember._internal.context.runtime import EmberContext

        return EmberContext.current().get_credential("azure_openai")


def _coerce_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload)


def _config_value(field: str) -> Any:
    from ember._internal.context.runtime import EmberContext

    value = EmberContext.current().get_config(f"providers.azure_openai.{field}")
    return value


def _config_str(field: str) -> Optional[str]:
    raw = _config_value(field)
    if raw is None:
        return None
    if isinstance(raw, str):
        trimmed = raw.strip()
        return trimmed or None
    raise TypeError(f"providers.azure_openai.{field} must be a string, got {type(raw).__name__}")


def _config_list(field: str) -> tuple[str, ...]:
    raw = _config_value(field)
    if raw is None:
        return ()
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    raise TypeError(f"providers.azure_openai.{field} must be a sequence of strings")


def _config_bool(field: str, *, default: bool) -> bool:
    raw = _config_value(field)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise TypeError(f"providers.azure_openai.{field} must be a boolean")


def _build_messages(
    prompt: str,
    system: Optional[str],
    context: Optional[Any],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if system:
        messages.append({"role": "system", "content": system})

    if context is not None:
        if isinstance(context, Mapping):
            messages.append(_coerce_message(context))
        elif isinstance(context, Sequence) and not isinstance(context, (str, bytes)):
            for entry in context:
                if not isinstance(entry, Mapping):
                    raise TypeError("Azure context sequences must contain mapping objects")
                messages.append(_coerce_message(entry))
        elif isinstance(context, str):
            messages.append({"role": "system", "content": context})
        else:
            raise TypeError("Azure context must be a string, mapping, or sequence of mappings")

    messages.append({"role": "user", "content": prompt})
    return messages


def _coerce_message(entry: Mapping[str, Any]) -> dict[str, Any]:
    role = str(entry.get("role") or "user")
    content = entry.get("content")
    return {"role": role, "content": content if content is not None else ""}


def _as_responses_turns(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            turns.append({"role": role, "content": content})
            continue
        text_value = "" if content is None else str(content)
        turns.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": text_value}],
            }
        )
    return turns


def _usage_from_payload(payload: Any) -> Optional[UsageStats]:
    if not payload:
        return None

    def _read_int(source: Any, name: str) -> int:
        value = getattr(source, name, None)
        if value is None and isinstance(source, Mapping):
            value = source.get(name)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    prompt_tokens = _read_int(payload, "prompt_tokens") or _read_int(payload, "input_tokens")
    completion_tokens = _read_int(payload, "completion_tokens") or _read_int(
        payload, "output_tokens"
    )
    total_tokens = _read_int(payload, "total_tokens")
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _extract_text_from_responses(resp: Any) -> str:
    if not resp:
        return ""

    if getattr(resp, "output_text", None):
        return cast(str, resp.output_text)

    output = getattr(resp, "output", None)
    if not output:
        return ""

    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not content and isinstance(item, Mapping):
            content = item.get("content")
        if not content:
            continue
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
    return "\n".join(parts)


def _responses_output(resp: Any) -> list[dict[str, Any]] | None:
    output = getattr(resp, "output", None)
    if not output:
        return None
    return [_serialize_openai_object(item) for item in output]


def _serialize_openai_object(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _serialize_openai_object(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_openai_object(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _serialize_openai_object(to_dict())
        except Exception:
            pass
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _serialize_openai_object(model_dump())
        except Exception:
            pass
    return str(value)


def create_azure_shard_class(
    name: str,
    *,
    endpoint: str,
    api_version: Optional[str] = None,
    responses_deployments: Optional[Sequence[str]] = None,
    default_use_responses: Optional[bool] = None,
) -> type[AzureOpenAIProvider]:
    """Create a concrete AzureOpenAIProvider subclass bound to a specific shard."""

    class AzureShard(AzureOpenAIProvider):
        def __init__(self, api_key: Optional[str] = None):
            super().__init__(
                api_key=api_key,
                endpoint=endpoint,
                api_version=api_version,
                responses_deployments=responses_deployments,
                default_use_responses=default_use_responses,
            )

    AzureShard.__name__ = f"AzureOpenAIProvider_{name}"
    AzureShard.__qualname__ = AzureShard.__name__
    AzureShard.__doc__ = f"Azure OpenAI provider bound to shard '{name}' ({endpoint})."
    return AzureShard
