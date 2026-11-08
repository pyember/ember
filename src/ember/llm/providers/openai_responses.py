"""OpenAI Responses: the request body and the response, both ways.

This module is the only code that reads or writes Responses shapes. The
``openai`` provider uses it for models whose catalog entry says
``api = "openai_responses"``.

A streamed response ends with ``response.completed``, ``response.incomplete``
or ``response.failed``, each carrying the whole response object, so the
stream is decoded into deltas for the caller and the final `Decoded` comes
from the same walker that decodes a non-streamed response.

Wire reference, read 2026-11-07:
https://developers.openai.com/api/reference/resources/responses/methods/create
https://developers.openai.com/api/reference/resources/responses/streaming-events
"""

from collections.abc import Iterable, Iterator, Mapping

from ember.errors import ErrorKind
from ember.llm.provider import Decoded, StreamItem
from ember.llm.providers.openai_chat import arguments_json, image_url
from ember.llm.providers.sse import SseEvent
from ember.llm.providers.wire import (
    AssistantPart,
    Body,
    Codec,
    WireError,
    array,
    as_json,
    assemble,
    complete_arguments,
    count,
    identifier,
    json_object,
    malformed,
    merge_extensions,
    parse_event_data,
    replays,
    stream_cut,
    string,
)
from ember.llm.request import (
    ImagePart,
    JsonValue,
    Message,
    Part,
    ReasoningPart,
    Request,
    Sampling,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
)
from ember.llm.result import (
    FinishReason,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    Usage,
    UsageEvent,
)
from ember.llm.spec import ModelSpec

__all__ = ["CODEC", "decode", "decode_events", "encode"]

_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_SCHEMA_NAME = "output"

_INCOMPLETE_REASONS: Mapping[str, FinishReason] = {
    "max_output_tokens": FinishReason.LENGTH,
    "max_messages": FinishReason.LENGTH,
    "steered": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}

# ``response.failed`` codes that mean a policy blocked the output; content
# filtering is a finish reason, not an error.
_POLICY_CODES = frozenset(
    {"bio_policy", "misalignment_policy_violation", "image_content_policy_violation"}
)
_RETRYABLE_FAILURES: Mapping[str, ErrorKind] = {
    "server_error": ErrorKind.SERVER,
    "vector_store_timeout": ErrorKind.SERVER,
    "rate_limit_exceeded": ErrorKind.RATE_LIMITED,
}

_TEXT_DELTAS = frozenset({"response.output_text.delta", "response.refusal.delta"})
_REASONING_DELTAS = frozenset(
    {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}
)
_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed"})


def encode(spec: ModelSpec, request: Request, stream: bool) -> Body:
    """Builds the request body.

    Raises:
        ValueError: The request sets something Responses does not have
            (``stop``, ``seed``), or an extension collides with a field ember
            sets.
    """
    body: Body = {"model": spec.wire_id, "input": _input(spec, request.messages)}
    if request.system is not None:
        body["instructions"] = request.system
    body.update(_sampling(request.sampling))
    if request.tools:
        body["tools"] = [_tool(tool) for tool in request.tools]
        body["tool_choice"] = _tool_choice(request.tool_choice)
    if request.output_schema is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": _SCHEMA_NAME,
                "schema": request.output_schema,
                "strict": True,
            }
        }
    if stream:
        body["stream"] = True
    return merge_extensions(body, request.extensions)


def _sampling(sampling: Sampling) -> Body:
    if sampling.stop:
        raise ValueError("the Responses API has no stop sequences")
    if sampling.seed is not None:
        raise ValueError("the Responses API has no seed")
    fields: Body = {}
    if sampling.temperature is not None:
        fields["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        fields["top_p"] = sampling.top_p
    if sampling.max_output_tokens is not None:
        fields["max_output_tokens"] = sampling.max_output_tokens
    if sampling.reasoning_effort is not None:
        if sampling.reasoning_effort not in _EFFORTS:
            raise ValueError(
                f"Responses reasoning.effort must be one of {sorted(_EFFORTS)}, "
                f"got {sampling.reasoning_effort!r}"
            )
        fields["reasoning"] = {"effort": sampling.reasoning_effort}
    return fields


def _input(spec: ModelSpec, messages: tuple[Message, ...]) -> list[JsonValue]:
    items: list[JsonValue] = []
    for message in messages:
        if message.role == "user":
            items.append({"role": "user", "content": _user_content(message.parts)})
        elif message.role == "assistant":
            items.extend(_assistant_items(spec, message.parts))
        else:
            items.extend(_tool_output(part) for part in message.parts)
    return items


def _user_content(parts: tuple[Part, ...]) -> JsonValue:
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    content: list[JsonValue] = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "input_image", "image_url": image_url(part), "detail": "auto"})
    return content


def _assistant_items(spec: ModelSpec, parts: tuple[Part, ...]) -> Iterator[JsonValue]:
    """Encodes an assistant turn as input items, in order.

    Reasoning from this wire protocol is replayed as the item it came from;
    reasoning from other protocols cannot be read here and is left out.
    """
    for part in parts:
        if isinstance(part, TextPart):
            yield {"role": "assistant", "content": part.text}
        elif isinstance(part, ToolCallPart):
            yield {
                "type": "function_call",
                "call_id": part.id,
                "name": part.name,
                "arguments": arguments_json(part.arguments),
            }
        elif isinstance(part, ReasoningPart) and replays(part, spec):
            yield part.state


def _tool_output(part: Part) -> JsonValue:
    assert isinstance(part, ToolResultPart)  # Tool turns hold only tool results.
    if part.is_error:
        raise ValueError(
            "the Responses API has no field for a failed tool call; describe the failure "
            "in the result content and set is_error=False"
        )
    return {"type": "function_call_output", "call_id": part.call_id, "output": part.content}


def _tool(tool: ToolSpec) -> JsonValue:
    # strict=False matches Chat Completions' default, so a tool behaves the
    # same on both OpenAI wire protocols.
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


def _tool_choice(choice: ToolChoice) -> JsonValue:
    if choice in ("auto", "none", "required"):
        return choice
    return {"type": "function", "name": choice}


def decode(spec: ModelSpec, body: object, request_id: str | None) -> Decoded:
    """Decodes a complete response object.

    Raises:
        WireError: The response failed, was cancelled, or is malformed.
    """
    return _decode_response(json_object(body, "response"), request_id, spec.ref.provider)


def _decode_response(
    response: Mapping[str, object], request_id: str | None, provider: str
) -> Decoded:
    output = array(response.get("output"), "output")
    items = [json_object(item, "output[]") for item in output]
    finish = _finish(response, items)
    parts: list[AssistantPart] = []
    for index, item in enumerate(items):
        parts.extend(_item_parts(item, index, finish, provider))
    return assemble(parts, finish, usage(response.get("usage")), request_id)


def _finish(response: Mapping[str, object], items: list[Mapping[str, object]]) -> FinishReason:
    status = response.get("status")
    if status == "failed":
        return _failure(response.get("error"))
    if status == "cancelled":
        raise WireError(ErrorKind.CANCELLED, "the provider cancelled the response")
    if status == "incomplete":
        details = json_object(response.get("incomplete_details"), "incomplete_details")
        reason = details.get("reason")
        if not isinstance(reason, str) or reason not in _INCOMPLETE_REASONS:
            raise malformed(f"unknown incomplete reason {identifier(reason) or 'missing'}")
        return _INCOMPLETE_REASONS[reason]
    if status != "completed":
        raise malformed(f"unexpected status {identifier(status) or 'missing'}")
    if any(_is_refusal(item) for item in items):
        return FinishReason.REFUSAL
    if any(item.get("type") == "function_call" for item in items):
        return FinishReason.TOOL_CALLS
    return FinishReason.STOP


def _is_refusal(item: Mapping[str, object]) -> bool:
    if item.get("type") != "message":
        return False
    content = array(item.get("content"), "message.content")
    return any(json_object(c, "message.content[]").get("type") == "refusal" for c in content)


def _failure(error: object) -> FinishReason:
    """Maps a failed response: a policy block is content filtering, the rest raise."""
    fields = json_object(error, "error")
    code = fields.get("code")
    if code in _POLICY_CODES:
        return FinishReason.CONTENT_FILTER
    label = identifier(code)
    kind = _RETRYABLE_FAILURES.get(label or "", ErrorKind.BAD_REQUEST)
    reason = "the provider reported the response failed"
    raise WireError(kind, f"{reason}; provider error {label}" if label else reason)


def _item_parts(
    item: Mapping[str, object], index: int, finish: FinishReason, provider: str
) -> Iterator[AssistantPart]:
    """Maps one output item; built-in tool items (search, code) carry no part."""
    kind = item.get("type")
    if kind == "message":
        for raw in array(item.get("content"), f"output[{index}].content"):
            content = json_object(raw, f"output[{index}].content[]")
            if content.get("type") == "output_text":
                yield TextPart(string(content.get("text"), "output_text.text"))
            elif content.get("type") == "refusal":
                yield TextPart(string(content.get("refusal"), "refusal.refusal"))
    elif kind == "reasoning":
        yield ReasoningPart(
            text=_reasoning_text(item),
            api="openai_responses",
            provider=provider,
            state=as_json(item),
        )
    elif kind == "function_call":
        call = _function_call(item, index, finish)
        if call is not None:
            yield call


def _reasoning_text(item: Mapping[str, object]) -> str:
    pieces = []
    for key in ("summary", "content"):
        entries = item.get(key)
        for raw in array(entries, f"reasoning.{key}") if entries is not None else []:
            pieces.append(string(json_object(raw, f"reasoning.{key}[]").get("text"), "text"))
    return "".join(pieces)


def _function_call(
    item: Mapping[str, object], index: int, finish: FinishReason
) -> ToolCallPart | None:
    arguments = complete_arguments(
        string(item.get("arguments"), f"output[{index}].arguments"), finish, index
    )
    if arguments is None:
        return None
    return ToolCallPart(
        id=string(item.get("call_id"), f"output[{index}].call_id"),
        name=string(item.get("name"), f"output[{index}].name"),
        arguments=arguments,
    )


def usage(value: object) -> Usage:
    """Maps ``usage``; ``input_tokens`` includes cached and cache-written tokens."""
    if value is None:
        return Usage(measured=False)
    fields = json_object(value, "usage")
    prompt = count(fields.get("input_tokens"), "usage.input_tokens")
    output = count(fields.get("output_tokens"), "usage.output_tokens")
    input_details = json_object(fields.get("input_tokens_details", {}), "input_tokens_details")
    output_details = json_object(fields.get("output_tokens_details", {}), "output_tokens_details")
    cache_read = count(input_details.get("cached_tokens", 0), "usage.cached_tokens")
    cache_write = count(input_details.get("cache_write_tokens", 0), "usage.cache_write_tokens")
    reasoning = count(output_details.get("reasoning_tokens", 0), "usage.reasoning_tokens")
    if cache_read + cache_write > prompt or reasoning > output:
        raise malformed("usage details exceed their totals")
    return Usage(
        input=prompt - cache_read - cache_write,
        cache_read=cache_read,
        cache_write=cache_write,
        output=output,
        reasoning=reasoning,
    )


def decode_events(
    spec: ModelSpec, events: Iterable[SseEvent], request_id: str | None
) -> Iterator[StreamItem]:
    """Decodes a streamed response.

    Raises:
        WireError: The stream reports an error, the response failed, or the
            stream ends before a terminal event.
    """
    call_indexes: dict[int, int] = {}
    for event in events:
        data = parse_event_data(event, "stream event")
        kind = data.get("type")
        if kind in _TEXT_DELTAS:
            yield TextDelta(string(data.get("delta"), f"{kind}.delta"))
        elif kind in _REASONING_DELTAS:
            yield ReasoningDelta(string(data.get("delta"), f"{kind}.delta"))
        elif kind == "response.output_item.added":
            yield from _call_started(data, call_indexes)
        elif kind == "response.function_call_arguments.delta":
            yield _call_arguments(data, call_indexes)
        elif kind == "error":
            raise _stream_error(data)
        elif kind in _TERMINAL_EVENTS:
            response = json_object(data.get("response"), "terminal event response")
            decoded = _decode_response(response, request_id, spec.ref.provider)
            if decoded.usage.measured:
                yield UsageEvent(decoded.usage)
            yield decoded
            return
    raise stream_cut("response.completed")


def _call_started(data: Mapping[str, object], indexes: dict[int, int]) -> Iterator[StreamItem]:
    item = json_object(data.get("item"), "output_item.added.item")
    if item.get("type") != "function_call":
        return
    output_index = count(data.get("output_index"), "output_item.added.output_index")
    indexes[output_index] = len(indexes)
    arguments = item.get("arguments")
    yield ToolCallDelta(
        index=indexes[output_index],
        id=string(item.get("call_id"), "function_call.call_id"),
        name=string(item.get("name"), "function_call.name"),
        arguments_json="" if arguments is None else string(arguments, "function_call.arguments"),
    )


def _call_arguments(data: Mapping[str, object], indexes: dict[int, int]) -> ToolCallDelta:
    output_index = count(data.get("output_index"), "function_call_arguments.output_index")
    if output_index not in indexes:
        raise malformed("function call arguments arrived before the call")
    return ToolCallDelta(
        index=indexes[output_index],
        id=None,
        name=None,
        arguments_json=string(data.get("delta"), "function_call_arguments.delta"),
    )


def _stream_error(data: Mapping[str, object]) -> WireError:
    label = identifier(data.get("code"))
    kind = _RETRYABLE_FAILURES.get(label or "", ErrorKind.SERVER)
    reason = "the provider reported an error in the stream"
    return WireError(kind, f"{reason}; provider error {label}" if label else reason)


CODEC = Codec(encode=encode, decode=decode, decode_events=decode_events)
