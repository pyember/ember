"""Shared helpers for provider HTTP wrappers.

These functions let individual provider classes focus on shaping requests
while reusing common response parsing and validation logic across adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ember._internal.exceptions import ProviderAPIError
from ember.models.catalog.parameters import get_responses_policy
from ember.models.schemas import UsageStats

ResponsesJSON = Mapping[str, Any]


def flatten_content(content: Any) -> str:
    """Normalise assistant content payloads into a flat string."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
                elif text_val is not None:
                    parts.append(str(text_val))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if content is None:
        return ""

    return str(content)


def parse_usage_payload(usage_payload: Any, *, model: str) -> UsageStats | None:
    """Convert a provider usage payload into ``UsageStats``.

    Raises:
        ProviderAPIError: when the structure is malformed.
    """

    if usage_payload is None:
        return None
    if not isinstance(usage_payload, dict):
        raise ProviderAPIError(
            "Usage payload must be an object.",
            context={"model": model, "error_type": "invalid_response"},
        )
    try:
        prompt_tokens = int(usage_payload["prompt_tokens"])
        completion_tokens = int(usage_payload["completion_tokens"])
        total_tokens = int(usage_payload["total_tokens"])
        cost_value = usage_payload.get("cost_usd")
        actual_cost = None if cost_value is None else float(cost_value)
    except KeyError as exc:
        raise ProviderAPIError(
            "Usage payload missing required fields.",
            context={"model": model, "error_type": "invalid_response"},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ProviderAPIError(
            "Usage payload contained non-numeric values.",
            context={"model": model, "error_type": "invalid_response"},
        ) from exc

    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        actual_cost_usd=actual_cost,
    )


def extract_first_choice_text(data: Mapping[str, Any], *, model: str) -> str:
    """Pull the first choice text out of a chat completion response."""

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderAPIError(
            "Gateway response did not include choices.",
            context={"model": model, "error_type": "invalid_response"},
        )

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ProviderAPIError(
            "Gateway response choices must be objects.",
            context={"model": model, "error_type": "invalid_response"},
        )

    message = first_choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise ProviderAPIError(
            "Gateway response missing message content.",
            context={"model": model, "error_type": "invalid_response"},
        )

    return flatten_content(message.get("content"))


@dataclass(frozen=True)
class ResponsesContentPart:
    """Single content element inside an input/output message."""

    type: str
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class ResponsesMessage:
    """User/assistant/tool message preserved for the Responses API."""

    role: str
    content: Sequence[ResponsesContentPart]
    status: str | None = None


@dataclass(frozen=True)
class ResponsesRequest:
    """Validated Responses request payload."""

    model: str
    input: Sequence[ResponsesMessage] | str | None
    instructions: str | None
    messages: Sequence[ResponsesMessage]
    metadata: Mapping[str, Any] | None
    response_format: Mapping[str, Any] | None
    tools: Sequence[Mapping[str, Any]] | None
    tool_choice: Mapping[str, Any] | str | None
    parallel_tool_calls: bool | None
    stream: bool
    max_output_tokens: int | None
    temperature: float | None
    top_p: float | None
    stop_sequences: tuple[str, ...]
    reasoning: Mapping[str, Any] | None
    previous_response_id: str | None
    user: str | None
    session_id: str | None
    idempotency_key: str | None
    policy: Mapping[str, Any] | None
    original_payload: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class PromptComponents:
    """Lightweight representation for providers that expect system/context/prompt."""

    prompt: str
    system: str | None
    context: tuple[Mapping[str, str], ...]


def parse_responses_request(payload: Mapping[str, Any]) -> ResponsesRequest:
    """Parse and validate a Responses payload supplied by clients."""

    data: MutableMapping[str, Any] = dict(payload)

    model_raw = data.get("model")
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise ValueError("Responses payload requires a non-empty 'model'")

    instructions = data.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("Responses 'instructions' must be a string")

    input_raw = data.get("input")
    if isinstance(input_raw, list):
        normalized_input: Sequence[ResponsesMessage] | str | None = _normalize_messages(input_raw)
    elif isinstance(input_raw, str):
        normalized_input = input_raw
    elif input_raw is None:
        normalized_input = None
    else:
        raise ValueError("Responses 'input' must be a string or list of messages")

    messages = _normalize_messages(
        data.get("messages") if isinstance(data.get("messages"), list) else None
    )

    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("Responses 'metadata' must be an object")

    response_format = data.get("response_format")
    if response_format is not None and not isinstance(response_format, Mapping):
        raise ValueError("Responses 'response_format' must be an object")

    tools = data.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError("Responses 'tools' must be an array")

    tool_choice = data.get("tool_choice")
    if tool_choice is not None and not isinstance(tool_choice, (str, Mapping)):
        raise ValueError("Responses 'tool_choice' must be a string or object")

    parallel_tool_calls = data.get("parallel_tool_calls")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise ValueError("Responses 'parallel_tool_calls' must be a boolean")

    stream = bool(data.get("stream", False))

    max_output_tokens = data.get("max_output_tokens")
    if max_output_tokens is not None:
        if not isinstance(max_output_tokens, int):
            raise ValueError("Responses 'max_output_tokens' must be an integer")

    temperature = data.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (float, int)):
            raise ValueError("Responses 'temperature' must be numeric")
        temperature = float(temperature)

    top_p = data.get("top_p")
    if top_p is not None:
        if not isinstance(top_p, (float, int)):
            raise ValueError("Responses 'top_p' must be numeric")
        top_p = float(top_p)

    stop_sequences = _collect_stop_sequences(data)

    reasoning = data.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, Mapping):
        raise ValueError("Responses 'reasoning' must be an object")

    previous_response_id = data.get("previous_response_id")
    if previous_response_id is not None and not isinstance(previous_response_id, str):
        raise ValueError("Responses 'previous_response_id' must be a string")

    user = data.get("user")
    if user is not None and not isinstance(user, str):
        raise ValueError("Responses 'user' must be a string")

    session_id = data.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise ValueError("Responses 'session_id' must be a string")

    idempotency_key = data.get("idempotency_key")
    if idempotency_key is not None and not isinstance(idempotency_key, str):
        raise ValueError("Responses 'idempotency_key' must be a string")

    policy = data.get("policy")
    if policy is not None and not isinstance(policy, Mapping):
        raise ValueError("Responses 'policy' must be an object")

    return ResponsesRequest(
        model=model_raw.strip(),
        input=normalized_input,
        instructions=instructions,
        messages=messages,
        metadata=metadata,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        stream=stream,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences,
        reasoning=reasoning,
        previous_response_id=previous_response_id,
        user=user,
        session_id=session_id,
        idempotency_key=idempotency_key,
        policy=policy,
        original_payload=dict(payload),
    )


def build_prompt_components(request: ResponsesRequest) -> PromptComponents:
    """Convert a ResponsesRequest into provider-friendly prompt pieces."""

    turns = _collect_turns(request)
    if not turns:
        return PromptComponents(prompt="", system=request.instructions, context=tuple())

    context_entries = tuple({"role": role, "content": text} for role, text in turns[:-1])
    prompt = turns[-1][1]
    return PromptComponents(prompt=prompt, system=request.instructions, context=context_entries)


def text_turns(request: ResponsesRequest) -> tuple[tuple[str, str], ...]:
    """Return role/text pairs extracted from a ResponsesRequest."""

    return tuple(_collect_turns(request))


def _as_content_part(raw: Mapping[str, Any]) -> ResponsesContentPart:
    part_type = str(raw.get("type") or "input_text")
    fields = {key: value for key, value in raw.items() if key != "type"}
    return ResponsesContentPart(type=part_type, fields=fields)


def _normalize_message(raw: Mapping[str, Any]) -> ResponsesMessage:
    role = str(raw.get("role") or "user")
    status = raw.get("status")
    if status is not None:
        status = str(status)

    content_raw = raw.get("content")
    if isinstance(content_raw, list):
        parts = [_as_content_part(part) for part in content_raw if isinstance(part, Mapping)]
    elif isinstance(content_raw, Mapping):
        parts = [_as_content_part(content_raw)]
    elif isinstance(content_raw, str):
        parts = [ResponsesContentPart(type="input_text", fields={"text": content_raw})]
    else:
        parts = []

    return ResponsesMessage(role=role, content=parts, status=status)


def _normalize_messages(items: Sequence[Any] | None) -> list[ResponsesMessage]:
    messages: list[ResponsesMessage] = []
    if items is None:
        return messages
    for item in items:
        if isinstance(item, Mapping):
            messages.append(_normalize_message(item))
    return messages


def _collect_stop_sequences(payload: Mapping[str, Any]) -> tuple[str, ...]:
    stop_sequences: list[str] = []

    stop_raw = payload.get("stop")
    if isinstance(stop_raw, str):
        stop_sequences.append(stop_raw)
    elif isinstance(stop_raw, list):
        for item in stop_raw:
            if not isinstance(item, str):
                raise ValueError("Responses 'stop' entries must be strings")
            stop_sequences.append(item)

    text_field = payload.get("text")
    if isinstance(text_field, Mapping):
        stop_value = text_field.get("stop")
        if isinstance(stop_value, str):
            stop_sequences.append(stop_value)
        elif isinstance(stop_value, list):
            for item in stop_value:
                if not isinstance(item, str):
                    raise ValueError("Responses 'text.stop' entries must be strings")
                stop_sequences.append(item)

    if not stop_sequences:
        return tuple()
    return tuple(dict.fromkeys(stop_sequences))


def _collect_turns(request: ResponsesRequest) -> list[tuple[str, str]]:
    raw_turns: list[ResponsesMessage] = []

    if isinstance(request.input, list):
        raw_turns.extend(request.input)
    elif isinstance(request.input, str):
        raw_turns.append(
            ResponsesMessage(
                role="user",
                content=[ResponsesContentPart(type="input_text", fields={"text": request.input})],
            )
        )

    if request.messages:
        raw_turns.extend(request.messages)

    turns: list[tuple[str, str]] = []
    for message in raw_turns:
        text = _extract_text(message)
        if text:
            turns.append((message.role, text))
    return turns


def _extract_text(message: ResponsesMessage) -> str:
    fragments: list[str] = []
    for part in message.content:
        if part.type in {"input_text", "output_text", "text"}:
            text_value = part.fields.get("text")
            if not isinstance(text_value, str):
                raise ValueError("Responses content part missing text field")
            fragments.append(text_value)
        else:
            raise ValueError(f"Unsupported Responses content type '{part.type}'")
    return "".join(fragments)


def sanitize_responses_payload(
    model: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Return a copy of ``payload`` with unsupported params removed."""

    requested_model = (model or str(payload.get("model") or "")).strip()
    if not requested_model:
        return dict(payload)

    from ember.models.providers import resolve_model_id  # local import to avoid cycles

    _, model_id = resolve_model_id(requested_model)
    policy = get_responses_policy(model_id)
    sanitized: dict[str, object] = dict(payload)
    sanitized["model"] = model_id

    if not policy.allow_temperature:
        sanitized.pop("temperature", None)
    if not policy.allow_top_p:
        sanitized.pop("top_p", None)
    if not policy.allow_reasoning:
        sanitized.pop("reasoning", None)

    return sanitized


__all__ = [
    "extract_first_choice_text",
    "flatten_content",
    "parse_usage_payload",
    "PromptComponents",
    "ResponsesContentPart",
    "ResponsesMessage",
    "ResponsesRequest",
    "build_prompt_components",
    "parse_responses_request",
    "sanitize_responses_payload",
    "text_turns",
]
