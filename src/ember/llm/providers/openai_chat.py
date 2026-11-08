"""OpenAI Chat Completions: the request body and the response, both ways.

This module is the only code that reads or writes Chat Completions shapes. The
``openai`` provider uses it for models whose catalog entry says
``api = "openai_chat"``, and every ``openai_compatible`` server uses it.

Wire reference, read 2026-11-07:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
"""

import base64
import json
from collections.abc import Iterable, Iterator, Mapping

from ember.errors import ErrorKind
from ember.llm.provider import Decoded, StreamItem
from ember.llm.providers.sse import SseEvent
from ember.llm.providers.wire import (
    AssistantPart,
    Body,
    Codec,
    StreamAccumulator,
    WireError,
    array,
    assemble,
    complete_arguments,
    count,
    identifier,
    json_object,
    malformed,
    merge_extensions,
    parse_event_data,
    stream_cut,
    string,
)
from ember.llm.request import (
    ImagePart,
    JsonValue,
    Message,
    Part,
    Request,
    Sampling,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
)
from ember.llm.result import FinishReason, Usage, UsageEvent
from ember.llm.spec import ModelSpec

__all__ = ["CODEC", "decode", "decode_events", "encode", "error_label"]

#: Values the ``reasoning_effort`` field accepts.
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})

_FINISH_REASONS: Mapping[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}

#: The ``response_format`` name ember gives the caller's output schema.
_SCHEMA_NAME = "output"


def encode(spec: ModelSpec, request: Request, stream: bool) -> Body:
    """Builds the request body.

    Raises:
        ValueError: The request uses something Chat Completions cannot carry,
            or an extension collides with a field ember sets.
    """
    body: Body = {"model": spec.wire_id, "messages": _messages(request)}
    body.update(_sampling(request.sampling))
    if request.tools:
        body["tools"] = [_tool(tool) for tool in request.tools]
        body["tool_choice"] = _tool_choice(request.tool_choice)
    if request.output_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": _SCHEMA_NAME, "schema": request.output_schema, "strict": True},
        }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return merge_extensions(body, request.extensions)


def _sampling(sampling: Sampling) -> Body:
    fields: Body = {}
    if sampling.temperature is not None:
        fields["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        fields["top_p"] = sampling.top_p
    if sampling.max_output_tokens is not None:
        fields["max_completion_tokens"] = sampling.max_output_tokens
    if sampling.stop:
        fields["stop"] = list(sampling.stop)
    if sampling.seed is not None:
        fields["seed"] = sampling.seed
    if sampling.reasoning_effort is not None:
        if sampling.reasoning_effort not in _EFFORTS:
            raise ValueError(
                f"Chat Completions reasoning_effort must be one of {sorted(_EFFORTS)}, "
                f"got {sampling.reasoning_effort!r}"
            )
        fields["reasoning_effort"] = sampling.reasoning_effort
    return fields


def _messages(request: Request) -> list[JsonValue]:
    messages: list[JsonValue] = []
    if request.system is not None:
        messages.append({"role": "system", "content": request.system})
    for message in request.messages:
        if message.role == "user":
            messages.append({"role": "user", "content": _user_content(message.parts)})
        elif message.role == "assistant":
            messages.append(_assistant(message))
        else:
            messages.extend(_tool_result(part) for part in message.parts)
    return messages


def _user_content(parts: tuple[Part, ...]) -> JsonValue:
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    content: list[JsonValue] = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "image_url", "image_url": {"url": image_url(part)}})
    return content


def image_url(part: ImagePart) -> str:
    """Returns the image's URL, or its bytes as a ``data:`` URL."""
    if part.url is not None:
        return part.url
    assert part.data is not None  # ImagePart holds exactly one of url and data.
    return f"data:{part.media_type};base64,{base64.b64encode(part.data).decode('ascii')}"


def _assistant(message: Message) -> JsonValue:
    """Encodes an assistant turn.

    Chat Completions has no way to replay reasoning, so `ReasoningPart`s are
    left out.
    """
    texts = [part.text for part in message.parts if isinstance(part, TextPart)]
    calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
    encoded: dict[str, JsonValue] = {"role": "assistant"}
    if texts:
        encoded["content"] = "".join(texts)
    if calls:
        encoded["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": arguments_json(call.arguments)},
            }
            for call in calls
        ]
    if not texts and not calls:
        encoded["content"] = ""
    return encoded


def arguments_json(arguments: Mapping[str, JsonValue]) -> str:
    """Serializes tool-call arguments the way the OpenAI APIs carry them."""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _tool_result(part: Part) -> JsonValue:
    assert isinstance(part, ToolResultPart)  # Tool turns hold only tool results.
    if part.is_error:
        raise ValueError(
            "Chat Completions has no field for a failed tool call; describe the failure "
            "in the result content and set is_error=False"
        )
    return {"role": "tool", "tool_call_id": part.call_id, "content": part.content}


def _tool(tool: ToolSpec) -> JsonValue:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _tool_choice(choice: ToolChoice) -> JsonValue:
    if choice in ("auto", "none", "required"):
        return choice
    return {"type": "function", "function": {"name": choice}}


def decode(spec: ModelSpec, body: object, request_id: str | None) -> Decoded:
    """Decodes a complete chat completion.

    Raises:
        WireError: The response is malformed or reports an unknown finish.
    """
    del spec  # Chat Completions returns no reasoning state to attribute.
    response = json_object(body, "response")
    choice = _only_choice(response)
    message = json_object(choice.get("message"), "choices[0].message")
    content = message.get("content")
    refusal = message.get("refusal")
    calls = message.get("tool_calls")
    raw_calls = array(calls, "message.tool_calls") if calls is not None else []
    finish = _finish(choice.get("finish_reason"), refusal is not None, bool(raw_calls))
    parts: list[AssistantPart] = []
    if content is not None:
        parts.append(TextPart(string(content, "message.content")))
    if refusal is not None:
        parts.append(TextPart(string(refusal, "message.refusal")))
    for index, raw in enumerate(raw_calls):
        call = _tool_call(raw, index, finish)
        if call is not None:
            parts.append(call)
    return assemble(parts, finish, usage(response.get("usage")), request_id)


def _only_choice(response: Mapping[str, object]) -> Mapping[str, object]:
    choices = array(response.get("choices"), "choices")
    if len(choices) != 1:
        raise malformed(f"expected one choice, got {len(choices)}")
    return json_object(choices[0], "choices[0]")


def _tool_call(raw: object, index: int, finish: FinishReason) -> ToolCallPart | None:
    call = json_object(raw, f"tool_calls[{index}]")
    if call.get("type") != "function":
        raise malformed(f"tool_calls[{index}] is not a function call")
    function = json_object(call.get("function"), f"tool_calls[{index}].function")
    arguments = complete_arguments(
        string(function.get("arguments"), f"tool_calls[{index}].function.arguments"),
        finish,
        index,
    )
    if arguments is None:
        return None
    return ToolCallPart(
        id=string(call.get("id"), f"tool_calls[{index}].id"),
        name=string(function.get("name"), f"tool_calls[{index}].function.name"),
        arguments=arguments,
    )


def _finish(raw: object, refused: bool, called_tools: bool) -> FinishReason:
    """Maps ``finish_reason``. A refusal or a tool call outranks ``stop``."""
    if not isinstance(raw, str) or raw not in _FINISH_REASONS:
        label = identifier(raw) or "missing"
        raise malformed(f"unknown finish_reason {label}")
    finish = _FINISH_REASONS[raw]
    if finish is FinishReason.STOP and refused:
        return FinishReason.REFUSAL
    if finish is FinishReason.STOP and called_tools:
        return FinishReason.TOOL_CALLS
    return finish


def usage(value: object) -> Usage:
    """Maps a ``usage`` object; no object means the provider measured nothing.

    ``prompt_tokens`` includes cached and cache-written tokens, and
    ``completion_tokens`` includes reasoning tokens.
    """
    if value is None:
        return Usage(measured=False)
    fields = json_object(value, "usage")
    prompt = count(fields.get("prompt_tokens"), "usage.prompt_tokens")
    output = count(fields.get("completion_tokens"), "usage.completion_tokens")
    prompt_details = _details(fields, "prompt_tokens_details")
    cache_read = count(prompt_details.get("cached_tokens", 0), "usage.cached_tokens")
    cache_write = count(prompt_details.get("cache_write_tokens", 0), "usage.cache_write_tokens")
    reasoning = count(
        _details(fields, "completion_tokens_details").get("reasoning_tokens", 0),
        "usage.reasoning_tokens",
    )
    if cache_read + cache_write > prompt or reasoning > output:
        raise malformed("usage details exceed their totals")
    return Usage(
        input=prompt - cache_read - cache_write,
        cache_read=cache_read,
        cache_write=cache_write,
        output=output,
        reasoning=reasoning,
    )


def _details(fields: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = fields.get(key)
    return {} if value is None else json_object(value, f"usage.{key}")


def decode_events(
    spec: ModelSpec, events: Iterable[SseEvent], request_id: str | None
) -> Iterator[StreamItem]:
    """Decodes a streamed chat completion.

    Raises:
        WireError: The stream reports an error, is malformed, or ends before
            ``data: [DONE]``.
    """
    acc = StreamAccumulator("openai_chat", spec.ref.provider)
    finish_raw: object = None
    refused = False
    measured: Usage | None = None
    for event in events:
        if event.data == "[DONE]":
            break
        chunk = parse_event_data(event, "stream chunk")
        if "error" in chunk:
            raise stream_error(chunk["error"])
        if chunk.get("usage") is not None:
            measured = usage(chunk["usage"])
        for choice in array(chunk.get("choices"), "chunk.choices"):
            choice_fields = json_object(choice, "chunk.choices[0]")
            delta = json_object(choice_fields.get("delta"), "chunk.choices[0].delta")
            refused = refused or delta.get("refusal") is not None
            yield from _delta_items(acc, delta)
            if choice_fields.get("finish_reason") is not None:
                finish_raw = choice_fields["finish_reason"]
    else:
        raise stream_cut("data: [DONE]")
    finish = _finish(finish_raw, refused, acc.has_tool_calls)
    final_usage = measured if measured is not None else Usage(measured=False)
    if measured is not None:
        yield UsageEvent(measured)
    yield assemble(acc.parts(finish), finish, final_usage, request_id)


def _delta_items(acc: StreamAccumulator, delta: Mapping[str, object]) -> Iterator[StreamItem]:
    content = delta.get("content")
    if content:
        yield acc.text("content", string(content, "delta.content"))
    refusal = delta.get("refusal")
    if refusal:
        yield acc.text("refusal", string(refusal, "delta.refusal"))
    calls = delta.get("tool_calls")
    for raw in array(calls, "delta.tool_calls") if calls is not None else []:
        call = json_object(raw, "delta.tool_calls[]")
        index = count(call.get("index"), "delta.tool_calls[].index")
        function = call.get("function")
        fields = json_object(function, "delta.tool_calls[].function") if function else {}
        call_id = call.get("id")
        name = fields.get("name")
        arguments = fields.get("arguments")
        yield acc.tool_call(
            ("call", index),
            id=None if call_id is None else string(call_id, "tool call id"),
            name=None if name is None else string(name, "tool call name"),
            arguments="" if arguments is None else string(arguments, "tool call arguments"),
        )


def stream_error(value: object) -> WireError:
    """Returns the error for an ``error`` object sent inside a stream."""
    label = error_label({"error": value})
    reason = "the provider reported an error in the stream"
    return WireError(ErrorKind.SERVER, f"{reason}; provider error {label}" if label else reason)


def error_label(body: object) -> str | None:
    """Returns the ``code`` or ``type`` of an OpenAI error body, if short."""
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    return identifier(error.get("code")) or identifier(error.get("type"))


CODEC = Codec(encode=encode, decode=decode, decode_events=decode_events)
