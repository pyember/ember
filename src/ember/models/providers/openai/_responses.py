from __future__ import annotations

import json
from collections.abc import Generator, Mapping, Sequence
from typing import Any

import httpcore
import httpx
import openai

from ember._internal.exceptions import ProviderAPIError
from ember.models.catalog.parameters import get_responses_policy
from ember.models.schemas import ChatResponse, UsageStats

from ._messages import as_responses_turns
from ._types import MessageList
from ._usage import usage_from_openai


class ResponsesHandler:
    def __init__(self, client: openai.OpenAI) -> None:
        self._client = client

    def complete(
        self,
        model: str,
        messages: MessageList,
        reasoning: dict[str, object] | None,
        text_cfg: dict[str, object] | None,
        options: dict[str, object],
    ) -> ChatResponse:
        payload = self._build_payload(model, messages, reasoning, text_cfg, options)

        response = self._create_response(model, payload)
        return self._finalize_complete(model, response)

    def complete_from_payload(self, payload: dict[str, object]) -> ChatResponse:
        model = str(payload.get("model", ""))
        normalized = self._prepare_payload(dict(payload), model=model)
        response = self._create_response(model, normalized)
        return self._finalize_complete(model, response)

    def stream(
        self,
        model: str,
        messages: MessageList,
        reasoning: dict[str, object] | None,
        text_cfg: dict[str, object] | None,
        options: dict[str, object],
    ) -> Generator[str, None, ChatResponse]:
        payload = self._build_payload(model, messages, reasoning, text_cfg, options)

        return self._stream_with_payload(model, payload)

    def stream_from_payload(
        self,
        payload: dict[str, object],
    ) -> Generator[str, None, ChatResponse]:
        model = str(payload.get("model", ""))
        normalized = self._prepare_payload(dict(payload), model=model)
        return self._stream_with_payload(model, normalized)

    def _create_response(self, model: str, payload: dict[str, object]) -> Any:
        try:
            return self._client.responses.create(**payload)  # type: ignore[call-overload,call-arg]
        except openai.AuthenticationError as exc:
            raise ProviderAPIError("Invalid OpenAI API key", context={"model": model}) from exc
        except openai.RateLimitError as exc:
            raise ProviderAPIError("OpenAI rate limit exceeded", context={"model": model}) from exc
        except openai.APIError as exc:
            raise ProviderAPIError(f"OpenAI API error: {exc}", context={"model": model}) from exc

    def _finalize_complete(self, model: str, response: Any) -> ChatResponse:
        text_output = getattr(response, "output_text", None) or ""
        response_id = getattr(response, "id", None)
        resolved_model = getattr(response, "model", None) or model
        output_items = _normalize_output_items(getattr(response, "output", None))
        usage = usage_from_openai(getattr(response, "usage", None))

        return ChatResponse(
            data=text_output,
            output=output_items,
            response_id=str(response_id) if response_id else None,
            usage=usage,
            model_id=str(resolved_model),
            raw_output=_to_serializable(response),
        )

    def _stream_with_payload(
        self,
        model: str,
        payload: dict[str, object],
    ) -> Generator[str, None, ChatResponse]:
        try:
            stream_manager = self._client.responses.stream(**payload)  # type: ignore[call-overload,call-arg]
        except openai.AuthenticationError as exc:
            raise ProviderAPIError("Invalid OpenAI API key", context={"model": model}) from exc
        except openai.RateLimitError as exc:
            raise ProviderAPIError("OpenAI rate limit exceeded", context={"model": model}) from exc
        except openai.APIError as exc:
            raise ProviderAPIError(f"OpenAI API error: {exc}", context={"model": model}) from exc

        def _generator() -> Generator[str, None, ChatResponse]:
            parts: list[str] = []
            accumulated_usage = UsageStats()

            stream: Any
            try:
                with stream_manager as stream:
                    for raw_event in stream:
                        event_payload = _to_serializable(raw_event)
                        if not isinstance(event_payload, dict):
                            chunk = str(event_payload)
                            if chunk:
                                parts.append(chunk)
                                yield chunk
                            continue

                        event_type = str(event_payload.get("type", ""))

                        if event_type == "response.error":
                            error = event_payload.get("error")
                            if error:
                                message = getattr(error, "message", "stream error")
                            else:
                                message = "stream error"
                            raise ProviderAPIError(
                                f"OpenAI streaming error: {message}",
                                context={"model": model},
                            )

                        delta_value = event_payload.get("delta")
                        if event_type == "response.output_text.delta":
                            if isinstance(delta_value, str) and delta_value:
                                parts.append(delta_value)
                            # Forward structured deltas as JSON without duplicating plain text.
                            elif isinstance(delta_value, dict):
                                parts.append(json.dumps(delta_value, ensure_ascii=False))

                        usage_delta = event_payload.get("usage_delta")
                        if isinstance(usage_delta, dict):
                            accumulated_usage.prompt_tokens += _read_int(
                                usage_delta, "prompt_tokens"
                            )
                            accumulated_usage.completion_tokens += _read_int(
                                usage_delta, "completion_tokens"
                            )
                            accumulated_usage.total_tokens += _read_int(usage_delta, "total_tokens")

                        yield json.dumps(event_payload, ensure_ascii=False)
            except (httpx.RemoteProtocolError, httpcore.RemoteProtocolError) as exc:
                raise ProviderAPIError(
                    "OpenAI streaming connection closed before completion",
                    context={"model": model},
                ) from exc

            final_response = stream.get_final_response()
            response_id = getattr(final_response, "id", None)
            resolved_model = getattr(final_response, "model", None) or model
            output_items = _normalize_output_items(getattr(final_response, "output", None))
            final_usage = usage_from_openai(getattr(final_response, "usage", None))
            if final_usage is None:
                final_usage = accumulated_usage

            return ChatResponse(
                data="".join(parts),
                output=output_items,
                response_id=str(response_id) if response_id else None,
                usage=final_usage,
                model_id=str(resolved_model),
                raw_output=_to_serializable(final_response),
            )

        return _generator()

    def _build_payload(
        self,
        model: str,
        messages: MessageList,
        reasoning: dict[str, object] | None,
        text_cfg: dict[str, object] | None,
        options: dict[str, object],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "input": as_responses_turns(messages),
        }

        if reasoning is not None:
            payload["reasoning"] = reasoning

        text_section = dict(text_cfg) if text_cfg is not None else None
        temperature_value = options.pop("temperature", None)
        top_p_value = options.pop("top_p", None)

        # Use policy system to determine which sampling params are allowed
        policy = get_responses_policy(model)

        if text_section is not None:
            if temperature_value is not None and policy.allow_temperature:
                text_section["temperature"] = _require_float(
                    "temperature",
                    temperature_value,
                    model=model,
                )
            if top_p_value is not None and policy.allow_top_p:
                text_section["top_p"] = _require_float(
                    "top_p",
                    top_p_value,
                    model=model,
                )
            payload["text"] = text_section
        else:
            if temperature_value is not None and policy.allow_temperature:
                payload["temperature"] = _require_float(
                    "temperature",
                    temperature_value,
                    model=model,
                )
            if top_p_value is not None and policy.allow_top_p:
                payload["top_p"] = _require_float(
                    "top_p",
                    top_p_value,
                    model=model,
                )

        if "logprobs" in options:
            raise ProviderAPIError(
                "logprobs is not supported on Responses API",
                context={"model": model},
            )

        if "max_tokens" in options and "max_output_tokens" not in payload:
            payload["max_output_tokens"] = _require_int(
                "max_tokens", options.pop("max_tokens"), model=model
            )
        if "max_output_tokens" in options:
            payload["max_output_tokens"] = _require_int(
                "max_output_tokens", options.pop("max_output_tokens"), model=model
            )

        options.pop("stream", None)

        allowed = {
            "max_output_tokens",
            "stop",
            "metadata",
            # Accept explicit system instructions for Responses calls.
            "instructions",
            "response_format",
            "tool_choice",
            "tools",
            "modalities",
            "user",
            "seed",
            "reasoning_budget",
            "parallel_tool_calls",
        }

        for key in list(options):
            if key in allowed:
                payload[key] = options.pop(key)
            # Reject unsupported parameters to match the Responses surface area.

        if options:
            raise ProviderAPIError(
                f"Unsupported parameters for Responses API models: {', '.join(sorted(options))}",
                context={"model": model},
            )

        return payload

    def _prepare_payload(self, payload: dict[str, object], *, model: str) -> dict[str, object]:
        payload.pop("stream", None)

        messages_value = payload.pop("messages", None)
        if messages_value is not None:
            if not isinstance(messages_value, Sequence) or isinstance(messages_value, (str, bytes)):
                raise ProviderAPIError(
                    "Responses payload 'messages' must be a sequence of message objects",
                    context={"model": model},
                )
            normalized_messages: list[dict[str, object]] = []
            for index, message in enumerate(messages_value):
                if not isinstance(message, Mapping):
                    raise ProviderAPIError(
                        f"Responses payload 'messages' entry {index} must be a mapping",
                        context={"model": model},
                    )
                normalized_messages.append(dict(message))
            try:
                turns = as_responses_turns(normalized_messages)
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderAPIError(
                    "Failed to normalize Responses payload messages for OpenAI provider",
                    context={"model": model},
                ) from exc

            existing_input = payload.get("input")
            if existing_input is None:
                payload["input"] = turns
            else:
                if not isinstance(existing_input, Sequence) or isinstance(
                    existing_input, (str, bytes)
                ):
                    raise ProviderAPIError(
                        "Responses payload 'input' must be a sequence when provided",
                        context={"model": model},
                    )
                if any(not isinstance(item, Mapping) for item in existing_input):
                    raise ProviderAPIError(
                        "Responses payload 'input' entries must be mappings",
                        context={"model": model},
                    )
                merged_input = [dict(item) for item in existing_input]
                merged_input.extend(turns)
                payload["input"] = merged_input

        text_section = payload.get("text")
        if text_section is not None and not isinstance(text_section, Mapping):
            raise ProviderAPIError(
                "Responses payload 'text' section must be a mapping",
                context={"model": model},
            )
        text_payload: dict[str, object] | None = dict(text_section) if text_section else None

        temperature_value = payload.pop("temperature", None)
        top_p_value = payload.pop("top_p", None)

        if text_payload is not None:
            if temperature_value is not None:
                text_payload["temperature"] = _require_float(
                    "temperature",
                    temperature_value,
                    model=model,
                )
            if top_p_value is not None:
                text_payload["top_p"] = _require_float(
                    "top_p",
                    top_p_value,
                    model=model,
                )
            payload["text"] = text_payload
        else:
            if temperature_value is not None:
                payload["temperature"] = _require_float(
                    "temperature",
                    temperature_value,
                    model=model,
                )
            if top_p_value is not None:
                payload["top_p"] = _require_float(
                    "top_p",
                    top_p_value,
                    model=model,
                )

        if "logprobs" in payload:
            payload.pop("logprobs", None)
            raise ProviderAPIError(
                "logprobs is not supported on Responses API",
                context={"model": model},
            )

        if "max_tokens" in payload and "max_output_tokens" in payload:
            raise ProviderAPIError(
                "Specify only one of 'max_tokens' or 'max_output_tokens' for Responses payloads",
                context={"model": model},
            )

        if "max_tokens" in payload:
            max_tokens_value = payload.pop("max_tokens")
            if "max_output_tokens" not in payload:
                payload["max_output_tokens"] = _require_int(
                    "max_tokens",
                    max_tokens_value,
                    model=model,
                )

        return {key: value for key, value in payload.items() if value is not None}


def _normalize_output_items(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    serializable = _to_serializable(value)
    if isinstance(serializable, list):
        return serializable
    return None


def _read_int(source: Any, name: str) -> int:
    if isinstance(source, dict):
        value = source.get(name)
    else:
        value = getattr(source, name, None)
    if value in (None, "", 0):
        return 0
    if not isinstance(value, (int, str, float)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _require_float(name: str, value: object, *, model: str) -> float:
    if not isinstance(value, (int, str, float)):
        raise ProviderAPIError(
            f"{name} must be numeric",
            context={"model": model, "parameter": name},
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderAPIError(
            f"{name} must be numeric",
            context={"model": model, "parameter": name},
        ) from exc


def _require_int(name: str, value: object, *, model: str) -> int:
    if not isinstance(value, (int, str, float)):
        raise ProviderAPIError(
            f"{name} must be an integer",
            context={"model": model, "parameter": name},
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProviderAPIError(
            f"{name} must be an integer",
            context={"model": model, "parameter": name},
        ) from exc


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_serializable(item) for item in value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_serializable(to_dict())
        except Exception:
            pass
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_serializable(model_dump(mode="json", warnings=False))
        except TypeError:
            try:
                return _to_serializable(model_dump())
            except Exception:
                pass
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {k: _to_serializable(v) for k, v in vars(value).items()}
        except Exception:
            pass
    return str(value)
