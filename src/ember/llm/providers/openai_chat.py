"""OpenAI Chat Completions: the streaming request body and its chunks.

This module is the only code that reads or writes Chat Completions shapes. The
``openai`` provider uses it for models whose catalog entry says
``api = "openai_chat"``, and every ``openai_compatible`` server uses it.
Requests always stream and ask for usage with
``stream_options.include_usage``, so a compatible server must support
streaming; one that sends no usage chunk yields unmeasured usage.

Wire reference, read 2026-11-07:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
"""

from collections.abc import Iterable, Iterator, Mapping

from ember.llm.provider import StreamItem
from ember.llm.providers import openai_platform
from ember.llm.providers.openai_platform import EFFORTS, SCHEMA_NAME, UsageFields
from ember.llm.providers.sse import SseEvent
from ember.llm.providers.wire import (
    Body,
    Codec,
    StreamAccumulator,
    array,
    assemble,
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

__all__ = ["CODEC", "decode_events", "encode"]

_FINISH_REASONS: Mapping[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}

_USAGE = UsageFields(prompt="prompt_tokens", output="completion_tokens")


def encode(spec: ModelSpec, request: Request) -> Body:
    """Builds the streaming request body.

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
            "json_schema": {"name": SCHEMA_NAME, "schema": request.output_schema, "strict": True},
        }
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
        if sampling.reasoning_effort not in EFFORTS:
            raise ValueError(
                f"Chat Completions reasoning_effort must be one of {sorted(EFFORTS)}, "
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
            url = openai_platform.image_url(part)
            content.append({"type": "image_url", "image_url": {"url": url}})
    return content


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
                "function": {
                    "name": call.name,
                    "arguments": openai_platform.arguments_json(call.arguments),
                },
            }
            for call in calls
        ]
    if not texts and not calls:
        encoded["content"] = ""
    return encoded


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


def decode_events(
    spec: ModelSpec, request: Request, events: Iterable[SseEvent], request_id: str | None
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
            error = json_object(chunk["error"], "stream error")
            failure = openai_platform.error_failure(error, None)
            raise failure.wire_error("the provider reported an error in the stream")
        if chunk.get("usage") is not None:
            measured = openai_platform.usage(chunk["usage"], _USAGE)
        for choice in array(chunk.get("choices"), "chunk.choices"):
            fields = json_object(choice, "chunk.choices[]")
            if fields.get("index", 0) != 0:
                raise malformed("a chunk carries a second choice")
            delta = json_object(fields.get("delta"), "chunk.choices[].delta")
            refused = refused or delta.get("refusal") is not None
            yield from _delta_items(acc, delta)
            if fields.get("finish_reason") is not None:
                finish_raw = fields["finish_reason"]
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
        call_id, name, arguments = call.get("id"), fields.get("name"), fields.get("arguments")
        yield acc.tool_call(
            ("call", index),
            id=None if call_id is None else string(call_id, "tool call id"),
            name=None if name is None else string(name, "tool call name"),
            arguments="" if arguments is None else string(arguments, "tool call arguments"),
        )


def _finish(raw: object, refused: bool, called_tools: bool) -> FinishReason:
    """Maps ``finish_reason``. A refusal or a tool call outranks ``stop``."""
    if not isinstance(raw, str) or raw not in _FINISH_REASONS:
        raise malformed(f"unknown finish_reason {identifier(raw) or 'missing'}")
    finish = _FINISH_REASONS[raw]
    if finish is FinishReason.STOP and refused:
        return FinishReason.REFUSAL
    if finish is FinishReason.STOP and called_tools:
        return FinishReason.TOOL_CALLS
    return finish


CODEC = Codec(encode=encode, decode_events=decode_events, classify=openai_platform.classify)
