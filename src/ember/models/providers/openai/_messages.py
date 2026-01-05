from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ember._internal.exceptions import ProviderAPIError

from ._types import MessageDict, MessageList


def _clone(value: object) -> object:
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def normalize_context(context: object) -> MessageList | None:
    if context is None:
        return None
    if isinstance(context, (str, bytes)) or not isinstance(context, Sequence):
        raise ProviderAPIError("context must be a sequence of messages")

    normalized: list[MessageDict] = []
    for index, item in enumerate(context):
        if not isinstance(item, Mapping):
            raise ProviderAPIError(f"context entry {index} must be a mapping")
        if "role" not in item or "content" not in item:
            raise ProviderAPIError(f"context entry {index} missing 'role' or 'content'")

        normalized_message: dict[str, Any] = dict(item)
        normalized_message["role"] = str(item["role"])
        normalized_message["content"] = _clone(item["content"])
        normalized.append(normalized_message)
    return normalized


def build_messages(
    prompt: str, system_prompt: str | None, context: MessageList | None
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": str(system_prompt)})

    if context is not None:
        for item in context:
            normalized: dict[str, Any] = dict(item)
            normalized["role"] = str(item["role"])
            normalized["content"] = _clone(item.get("content"))
            messages.append(normalized)

    messages.append({"role": "user", "content": prompt})
    return messages


def as_responses_turns(messages: MessageList) -> list[dict[str, object]]:
    turns: list[dict[str, object]] = []
    for message in messages:
        role = str(message["role"])
        content = message.get("content", "")
        turns.append({"role": role, "content": _coerce_responses_content(content)})
    return turns


def _coerce_responses_content(content: object) -> list[dict[str, object]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, bytes):
        return [{"type": "input_text", "text": content.decode("utf-8", errors="ignore")}]
    if isinstance(content, Mapping):
        return [_coerce_responses_part(content)]
    if isinstance(content, Sequence):
        parts: list[dict[str, object]] = []
        for part in content:
            if isinstance(part, Mapping):
                parts.append(_coerce_responses_part(part))
            elif isinstance(part, str):
                parts.append({"type": "input_text", "text": part})
            elif isinstance(part, bytes):
                parts.append(
                    {
                        "type": "input_text",
                        "text": part.decode("utf-8", errors="ignore"),
                    }
                )
            else:
                parts.append({"type": "input_text", "text": str(part)})
        return parts
    if content is None:
        return [{"type": "input_text", "text": ""}]
    return [{"type": "input_text", "text": str(content)}]


def _coerce_responses_part(part: Mapping[str, object]) -> dict[str, object]:
    converted = dict(part)
    part_type = str(converted.get("type") or "input_text")
    if part_type == "text":
        converted["type"] = "input_text"
    return converted
