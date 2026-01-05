from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any

import openai

from ember._internal.exceptions import ProviderAPIError
from ember.models.schemas import ChatResponse

from ._types import MessageList
from ._usage import usage_from_openai


class ChatCompletionsHandler:
    def __init__(self, client: openai.OpenAI) -> None:
        self._client = client

    def complete(
        self,
        model: str,
        messages: MessageList,
        options: dict[str, object],
    ) -> ChatResponse:
        params = self._build_params(model, messages, options, stream=False)

        try:
            response = self._client.chat.completions.create(**params)  # type: ignore[call-overload,call-arg]
        except openai.AuthenticationError as exc:
            raise ProviderAPIError("Invalid OpenAI API key", context={"model": model}) from exc
        except openai.RateLimitError as exc:
            raise ProviderAPIError("OpenAI rate limit exceeded", context={"model": model}) from exc
        except openai.APIError as exc:
            raise ProviderAPIError(f"OpenAI API error: {exc}", context={"model": model}) from exc
        except Exception as exc:  # pragma: no cover
            raise ProviderAPIError(f"Unexpected error: {exc}", context={"model": model}) from exc

        first_choice = response.choices[0]
        message = first_choice.message
        text_output = message.content or ""
        usage = usage_from_openai(response.usage)

        return ChatResponse(
            data=text_output,
            usage=usage,
            model_id=response.model or model,
            raw_output=response,
        )

    def stream(
        self,
        model: str,
        messages: MessageList,
        options: dict[str, object],
    ) -> Generator[str, None, ChatResponse]:
        params = self._build_params(model, messages, options, stream=True)

        try:
            stream = self._client.chat.completions.create(**params)  # type: ignore[call-overload,call-arg]
        except openai.AuthenticationError as exc:
            raise ProviderAPIError("Invalid OpenAI API key", context={"model": model}) from exc
        except openai.RateLimitError as exc:
            raise ProviderAPIError("OpenAI rate limit exceeded", context={"model": model}) from exc
        except openai.APIError as exc:
            raise ProviderAPIError(f"OpenAI API error: {exc}", context={"model": model}) from exc
        except Exception as exc:  # pragma: no cover
            raise ProviderAPIError(f"Unexpected error: {exc}", context={"model": model}) from exc

        def _generator() -> Generator[str, None, ChatResponse]:
            parts: list[str] = []
            final_usage = None

            for raw_event in stream:
                for chunk in _iter_chat_chunks(raw_event):
                    if chunk.text:
                        parts.append(chunk.text)
                        yield chunk.text
                    if chunk.serialised:
                        yield chunk.serialised

                usage_payload = getattr(raw_event, "usage", None)
                if usage_payload is None and isinstance(raw_event, dict):
                    usage_payload = raw_event.get("usage")
                if usage_payload is not None:
                    final_usage = usage_from_openai(usage_payload)

            text_output = "".join(parts)
            return ChatResponse(
                data=text_output,
                usage=final_usage,
                model_id=model,
                raw_output=None,
            )

        return _generator()

    def _build_params(
        self,
        model: str,
        messages: MessageList,
        options: dict[str, object],
        *,
        stream: bool,
    ) -> dict[str, object]:
        params: dict[str, object] = {"model": model, "messages": list(messages)}

        if "max_tokens" in options and "max_completion_tokens" in options:
            raise ProviderAPIError(
                "Specify only one of 'max_tokens' or 'max_completion_tokens' for OpenAI chat calls",
                context={"model": model},
            )

        if "max_tokens" in options:
            params["max_completion_tokens"] = options.pop("max_tokens")
        if "max_completion_tokens" in options:
            params["max_completion_tokens"] = options.pop("max_completion_tokens")

        allowed = {
            "temperature",
            "top_p",
            "max_completion_tokens",
            "stop",
            "logprobs",
            "logit_bias",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
            "tool_choice",
            "tools",
            "user",
            "seed",
            "functions",
            "function_call",
            "parallel_tool_calls",
            "metadata",
        }

        for key in list(options):
            if key in allowed:
                params[key] = options.pop(key)

        if options:
            raise ProviderAPIError(
                f"Unsupported parameters for OpenAI chat models: {', '.join(sorted(options))}",
                context={"model": model},
            )

        if stream:
            params["stream"] = True

        return params


@dataclass(frozen=True)
class _StreamChunk:
    text: str | None = None
    serialised: str | None = None


def _iter_chat_chunks(event: Any) -> Generator[_StreamChunk, None, None]:
    choices: Sequence[Any]
    if isinstance(event, dict):
        choices = event.get("choices", ())  # type: ignore[assignment]
    else:
        choices = getattr(event, "choices", ())

    for choice in choices or ():
        delta = choice
        if not isinstance(choice, dict):
            delta = getattr(choice, "delta", None)
        else:
            delta = choice.get("delta")

        content = _get_field(delta, "content")

        if isinstance(content, str):
            yield _StreamChunk(text=content)
        elif isinstance(content, list):
            for block in content:
                text = _get_field(block, "text")
                if isinstance(text, str) and text:
                    yield _StreamChunk(text=text)
        elif content:
            text_val = getattr(content, "text", None)
            if isinstance(text_val, str) and text_val:
                yield _StreamChunk(text=text_val)

        tool_calls = _get_field(delta, "tool_calls")
        if isinstance(tool_calls, Sequence):
            for call in tool_calls:
                serialised = _serialize_tool_call(call)
                if serialised:
                    yield _StreamChunk(serialised=serialised)


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _serialize_tool_call(call: Any) -> str | None:
    if call is None:
        return None
    payload: dict[str, Any] = {"type": "tool_call.delta"}
    payload["id"] = _get_field(call, "id")
    payload["index"] = _get_field(call, "index")
    function_payload = _get_field(call, "function")
    if function_payload is not None:
        payload["name"] = _get_field(function_payload, "name")
        payload["arguments_delta"] = _get_field(function_payload, "arguments")
    clean = {k: v for k, v in payload.items() if v is not None}
    if not clean:
        return None
    return json.dumps({"delta": clean})
