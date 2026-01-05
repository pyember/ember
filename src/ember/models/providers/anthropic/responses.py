"""Anthropic Responses API compatibility layer.

Converts OpenAI Responses-style payloads to Anthropic Messages format and
streams Anthropic events back as OpenAI Responses-style Server-Sent Events.

This enables clients speaking the Responses protocol to use Claude models
with minimal adaptation.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.utils import ResponsesRequest, parse_responses_request

_EFFORT_TO_BUDGET_TOKENS: Final[dict[str, int]] = {
    "low": 1024,
    "medium": 2048,
    "high": 3072,
}

_DEFAULT_MAX_TOKENS: Final[int] = 4096


@dataclass
class AnthropicParams:
    """Parameters for an Anthropic Messages API call."""

    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    system: list[dict[str, str]] | None = None
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: tuple[str, ...] = ()
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None


def validate_responses_payload(payload: Mapping[str, Any]) -> ResponsesRequest:
    """Validate and parse a Responses payload.

    Args:
        payload: Raw Responses request payload.

    Returns:
        Validated ResponsesRequest object.

    Raises:
        ProviderAPIError: When validation fails.
    """
    if not isinstance(payload, Mapping):
        raise ProviderAPIError("Responses payload must be a JSON object")
    try:
        return parse_responses_request(dict(payload))
    except ValueError as exc:
        raise ProviderAPIError(str(exc)) from exc


def build_anthropic_params(request: ResponsesRequest) -> AnthropicParams:
    """Convert a ResponsesRequest into Anthropic API parameters.

    Maps OpenAI Responses conventions (function tools, reasoning effort) to
    their Anthropic equivalents (tool definitions, thinking budgets).

    Args:
        request: Validated Responses request.

    Returns:
        AnthropicParams ready for Anthropic Messages API calls.
    """
    messages = _build_messages(request)
    system = _build_system(request.instructions)
    tools = _convert_tools(request.tools)
    tool_choice = _convert_tool_choice(
        request.tool_choice,
        disable_parallel=request.parallel_tool_calls is False,
    )
    thinking = _convert_reasoning(request.reasoning)

    return AnthropicParams(
        model=request.model,
        messages=messages,
        system=system,
        max_tokens=request.max_output_tokens or _DEFAULT_MAX_TOKENS,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=request.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        thinking=thinking,
    )


def build_completed_responses_payload(
    *,
    response_id: str,
    model: str,
    output_text: str,
    reasoning_summary: str | None,
    tool_calls: Sequence[dict[str, Any]] | None,
    usage: dict[str, int] | None,
) -> dict[str, Any]:
    """Build the final response.completed payload.

    Args:
        response_id: Unique response identifier.
        model: Model name.
        output_text: Accumulated text output.
        reasoning_summary: Accumulated reasoning/thinking text.
        tool_calls: Tool call requests if any.
        usage: Token usage statistics.

    Returns:
        Structured response payload matching Responses API format.
    """
    content_items: list[dict[str, Any]] = []

    if reasoning_summary:
        content_items.append({"type": "reasoning_summary", "text": reasoning_summary})
    if output_text:
        content_items.append({"type": "output_text", "text": output_text})
    if tool_calls:
        for call in tool_calls:
            content_items.append({
                "type": "tool_use",
                "id": call.get("id"),
                "name": call.get("name"),
                "arguments": call.get("arguments"),
            })

    # Output item format: role + content + status (no "type" field)
    output_item: dict[str, Any] = {
        "role": "assistant",
        "content": content_items,
        "status": "completed",
    }

    response: dict[str, Any] = {
        "id": response_id,
        "model": model,
        "status": "completed",
        "output": [output_item],
        "output_text": output_text,
    }

    if reasoning_summary:
        response["reasoning_summary"] = reasoning_summary

    if usage:
        response["usage"] = usage

    return response


def clone_messages(
    messages: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Deep clone a sequence of messages.

    Args:
        messages: Source messages to clone.

    Returns:
        New list with recursively cloned message dictionaries.
    """
    return [_deep_clone(dict(msg)) for msg in messages]


def stream_events_from_anthropic(
    stream: Iterable[Any],
    model: str,
    response_id: str | None = None,
) -> Iterable[str]:
    """Convert Anthropic streaming events to OpenAI Responses-style SSE.

    Translates content_block_start/delta/stop and message events from the
    Anthropic SDK into the event types expected by Responses API clients.

    Args:
        stream: Iterable of Anthropic streaming events.
        model: Model name for response payloads.
        response_id: Optional response ID; auto-generated if not provided.

    Yields:
        JSON-encoded event strings matching Responses API format.
    """
    if response_id is None:
        response_id = f"resp_{uuid.uuid4().hex[:24]}"

    yield json.dumps({
        "type": "response.created",
        "response": {"id": response_id, "model": model, "status": "in_progress"},
    })

    state = _StreamState()

    for event in stream:
        event_type = _get_event_type(event)

        if event_type == "content_block_start":
            content_block = _get_content_block(event)
            block_type = content_block.get("type") if content_block else None
            if block_type == "tool_use":
                tool_id = content_block.get("id", "")
                tool_name = content_block.get("name", "")
                state.start_tool_call(tool_id, tool_name)
                # Emit tool call created event
                yield json.dumps({
                    "type": "response.tool_call.created",
                    "call_id": tool_id,
                    "name": tool_name,
                    "tool_type": "function",
                })

        elif event_type == "content_block_delta":
            content_block = _get_content_block(event)
            delta = _get_delta(event)
            block_type = content_block.get("type") if content_block else None

            if block_type == "thinking":
                text = delta.get("text", "") if delta else ""
                if text:
                    state.reasoning_parts.append(text)
                    yield json.dumps({
                        "type": "response.reasoning_summary_text.delta",
                        "delta": text,
                    })

            elif block_type == "text":
                text = delta.get("text", "") if delta else ""
                if text:
                    state.text_parts.append(text)
                    yield json.dumps({
                        "type": "response.output_text.delta",
                        "delta": text,
                    })

            elif block_type == "tool_use":
                partial_json = delta.get("partial_json", "") if delta else ""
                if partial_json:
                    state.append_tool_json(partial_json)
                    # Emit tool call delta event
                    yield json.dumps({
                        "type": "response.tool_call.delta",
                        "call_id": state._current_tool_id,
                        "arguments_delta": partial_json,
                    })

        elif event_type == "content_block_stop":
            # Emit tool call completed if there's an active tool call
            if state._current_tool_id:
                yield json.dumps({
                    "type": "response.tool_call.completed",
                    "call_id": state._current_tool_id,
                })
            state.finalize_tool_call()

        elif event_type == "message_delta":
            # Extract usage from message_delta and emit response.delta
            usage = _get_usage(event)
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total = input_tokens + output_tokens
                state.usage = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                yield json.dumps({
                    "type": "response.delta",
                    "usage_delta": state.usage,
                })

        elif event_type == "message_stop":
            # Handle both dict and object events
            if isinstance(event, Mapping):
                stop_reason = event.get("stop_reason")
            else:
                stop_reason = getattr(event, "stop_reason", None)
            if stop_reason == "tool_use" and state.tool_calls:
                yield json.dumps({
                    "type": "response.required_action",
                    "required_action": {
                        "type": "submit_tool_outputs",
                        "submit_tool_outputs": {"tool_calls": state.tool_calls},
                    },
                })
            else:
                completed_event: dict[str, Any] = {
                    "type": "response.completed",
                    "response": build_completed_responses_payload(
                        response_id=response_id,
                        model=model,
                        output_text="".join(state.text_parts),
                        reasoning_summary="".join(state.reasoning_parts) or None,
                        tool_calls=state.tool_calls if state.tool_calls else None,
                        usage=state.usage,
                    ),
                }
                # Include usage at top level for easy access
                if state.usage:
                    completed_event["usage"] = state.usage
                yield json.dumps(completed_event)


class _StreamState:
    """Mutable state tracker for streaming event conversion."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.usage: dict[str, int] | None = None
        self._current_tool_id: str | None = None
        self._current_tool_name: str | None = None
        self._current_tool_json_parts: list[str] = []

    def start_tool_call(self, tool_id: str, tool_name: str) -> None:
        self._current_tool_id = tool_id
        self._current_tool_name = tool_name
        self._current_tool_json_parts = []

    def append_tool_json(self, partial: str) -> None:
        self._current_tool_json_parts.append(partial)

    def finalize_tool_call(self) -> None:
        if self._current_tool_id and self._current_tool_name:
            json_str = "".join(self._current_tool_json_parts)
            try:
                arguments = json.loads(json_str) if json_str else {}
            except json.JSONDecodeError:
                arguments = {}

            self.tool_calls.append({
                "id": self._current_tool_id,
                "name": self._current_tool_name,
                "type": "function",
                "arguments": arguments,
            })

        self._current_tool_id = None
        self._current_tool_name = None
        self._current_tool_json_parts = []


def _build_messages(request: ResponsesRequest) -> list[dict[str, Any]]:
    """Convert Responses input/messages to Anthropic message format."""
    messages: list[dict[str, Any]] = []

    if isinstance(request.input, list):
        for msg in request.input:
            text_parts = []
            for part in msg.content:
                if part.type in {"input_text", "output_text", "text"}:
                    text_val = part.fields.get("text")
                    if isinstance(text_val, str):
                        text_parts.append(text_val)
            if text_parts:
                messages.append({
                    "role": msg.role,
                    "content": [{"type": "text", "text": "".join(text_parts)}],
                })
    elif isinstance(request.input, str):
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": request.input}],
        })

    for msg in request.messages:
        text_parts = []
        for part in msg.content:
            if part.type in {"input_text", "output_text", "text"}:
                text_val = part.fields.get("text")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
        if text_parts:
            messages.append({
                "role": msg.role,
                "content": [{"type": "text", "text": "".join(text_parts)}],
            })

    return messages


def _build_system(instructions: str | None) -> list[dict[str, str]] | None:
    """Convert instructions to Anthropic system format."""
    if not instructions:
        return None
    return [{"type": "text", "text": instructions}]


def _convert_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI function-style tools to Anthropic format."""
    if not tools:
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        tool_type = tool.get("type")
        if tool_type == "function":
            func = tool.get("function", {})
            converted.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })
        else:
            converted.append(dict(tool))

    return converted if converted else None


def _convert_tool_choice(
    tool_choice: Mapping[str, Any] | str | None,
    *,
    disable_parallel: bool,
) -> dict[str, Any] | None:
    """Convert OpenAI tool_choice to Anthropic format."""
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto", "disable_parallel_tool_use": disable_parallel}
        if tool_choice == "none":
            return None
        if tool_choice == "required":
            return {"type": "any", "disable_parallel_tool_use": disable_parallel}
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "function":
        name = tool_choice.get("name") or tool_choice.get("function", {}).get("name")
        return {
            "type": "tool",
            "name": name,
            "disable_parallel_tool_use": disable_parallel,
        }

    return dict(tool_choice)


def _convert_reasoning(
    reasoning: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert reasoning effort to Anthropic thinking payload."""
    if reasoning is None:
        return None

    explicit_budget = reasoning.get("budget_tokens")
    if explicit_budget is not None:
        try:
            budget = int(explicit_budget)
        except (TypeError, ValueError):
            return None
        return {"type": "enabled", "budget_tokens": max(1024, budget)}

    effort = reasoning.get("effort")
    if effort is None:
        return None

    effort_key = str(effort).lower().strip()
    budget_tokens = _EFFORT_TO_BUDGET_TOKENS.get(effort_key)
    if budget_tokens is None:
        return None

    return {"type": "enabled", "budget_tokens": budget_tokens}


def _get_event_type(event: Any) -> str:
    """Extract event type from an Anthropic streaming event."""
    if isinstance(event, Mapping):
        return str(event.get("type", ""))
    return str(getattr(event, "type", ""))


def _get_content_block(event: Any) -> dict[str, Any] | None:
    """Extract content_block from an Anthropic streaming event."""
    if isinstance(event, Mapping):
        block = event.get("content_block")
    else:
        block = getattr(event, "content_block", None)

    if isinstance(block, Mapping):
        return dict(block)
    return None


def _get_delta(event: Any) -> dict[str, Any] | None:
    """Extract delta from an Anthropic streaming event."""
    if isinstance(event, Mapping):
        delta = event.get("delta")
    else:
        delta = getattr(event, "delta", None)

    if isinstance(delta, Mapping):
        return dict(delta)
    return None


def _get_usage(event: Any) -> dict[str, Any] | None:
    """Extract usage from an Anthropic streaming event."""
    if isinstance(event, Mapping):
        usage = event.get("usage")
    else:
        usage = getattr(event, "usage", None)

    if isinstance(usage, Mapping):
        return dict(usage)
    return None


def _deep_clone(obj: Any) -> Any:
    """Recursively clone nested dicts and lists."""
    if isinstance(obj, dict):
        return {k: _deep_clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_clone(item) for item in obj]
    return obj


__all__ = [
    "AnthropicParams",
    "build_anthropic_params",
    "build_completed_responses_payload",
    "clone_messages",
    "stream_events_from_anthropic",
    "validate_responses_payload",
]
