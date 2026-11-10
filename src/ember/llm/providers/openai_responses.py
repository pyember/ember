"""OpenAI Responses: the streaming request body and its events.

This module is the only code that reads or writes Responses shapes. The
``openai`` provider uses it for models whose catalog entry says
``api = "openai_responses"``.

A streamed response ends with ``response.completed``, ``response.incomplete``
or ``response.failed``, each carrying the whole response object; the deltas
before it go to the caller and the final `Decoded` is decoded from that
object.

For a reasoning model (one whose catalog entry lists reasoning efforts) the
request asks for ``reasoning.encrypted_content``, so the reasoning items in
`Result.message` can be replayed without relying on the provider storing
the response, including with ``store: false``.

Wire reference, read 2026-11-07:
https://developers.openai.com/api/reference/resources/responses/methods/create
https://developers.openai.com/api/reference/resources/responses/streaming-events
"""

from collections.abc import Iterable, Iterator, Mapping

from ember.errors import ErrorKind
from ember.llm.provider import Decoded, StreamItem
from ember.llm.providers import openai_platform
from ember.llm.providers.openai_platform import EFFORTS, SCHEMA_NAME, UsageFields
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
    UsageEvent,
)
from ember.llm.spec import ModelSpec

__all__ = ["CODEC", "decode_events", "encode"]

_USAGE = UsageFields(prompt="input_tokens", output="output_tokens")

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

_ENCRYPTED_REASONING = "reasoning.encrypted_content"

_TEXT_DELTAS = frozenset({"response.output_text.delta", "response.refusal.delta"})
_REASONING_DELTAS = frozenset(
    {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}
)
_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed"})


def encode(spec: ModelSpec, request: Request) -> Body:
    """Builds the streaming request body.

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
                "name": SCHEMA_NAME,
                "schema": request.output_schema,
                "strict": True,
            }
        }
    body["stream"] = True
    extensions = dict(request.extensions)
    include = _include(spec, extensions.pop("include", []))
    if include:
        body["include"] = include
    return merge_extensions(body, extensions)


def _include(spec: ModelSpec, requested: JsonValue) -> list[JsonValue]:
    """Returns ``include``: the caller's entries plus encrypted reasoning.

    A reasoning model's request always asks for ``reasoning.encrypted_content``;
    a caller's own ``include`` extension adds to that list instead of
    colliding with it, keeping the caller's order.
    """
    if not isinstance(requested, list) or not all(isinstance(e, str) for e in requested):
        raise ValueError("the include extension must be a list of strings")
    include: list[JsonValue] = list(dict.fromkeys(requested))
    if spec.reasoning_efforts and _ENCRYPTED_REASONING not in include:
        include.append(_ENCRYPTED_REASONING)
    return include


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
        if sampling.reasoning_effort not in EFFORTS:
            raise ValueError(
                f"Responses reasoning.effort must be one of {sorted(EFFORTS)}, "
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
            url = openai_platform.image_url(part)
            content.append({"type": "input_image", "image_url": url, "detail": "auto"})
    return content


def _assistant_items(spec: ModelSpec, parts: tuple[Part, ...]) -> Iterator[JsonValue]:
    """Encodes an assistant turn as input items, in order.

    Reasoning is replayed as the item it came from when it came from this
    provider and protocol; any other reasoning is left out.
    """
    for part in parts:
        if isinstance(part, TextPart):
            yield {"role": "assistant", "content": part.text}
        elif isinstance(part, ToolCallPart):
            yield {
                "type": "function_call",
                "call_id": part.id,
                "name": part.name,
                "arguments": openai_platform.arguments_json(part.arguments),
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


def decode_events(
    spec: ModelSpec, request: Request, events: Iterable[SseEvent], request_id: str | None
) -> Iterator[StreamItem]:
    """Decodes a streamed response.

    Raises:
        WireError: The stream reports an error, the response failed, or the
            stream ends before a terminal event.
    """
    call_indexes: dict[int, int] = {}
    for event in events:
        data = parse_event_data(event, "stream event")
        kind = _event_type(data)
        if kind in _TEXT_DELTAS:
            yield TextDelta(string(data.get("delta"), f"{kind}.delta"))
        elif kind in _REASONING_DELTAS:
            yield ReasoningDelta(string(data.get("delta"), f"{kind}.delta"))
        elif kind == "response.output_item.added":
            yield from _call_started(data, call_indexes)
        elif kind == "response.function_call_arguments.delta":
            yield _call_arguments(data, call_indexes)
        elif kind == "error":
            failure = openai_platform.error_failure({"code": data.get("code")}, None)
            raise failure.wire_error("the provider reported an error in the stream")
        elif kind in _TERMINAL_EVENTS:
            response = json_object(data.get("response"), f"{kind}.response")
            decoded = _decode_response(response, request_id, spec.ref.provider)
            if decoded.usage.measured:
                yield UsageEvent(decoded.usage)
            yield decoded
            return
    raise stream_cut("response.completed")


def _event_type(data: Mapping[str, object]) -> str:
    kind = data.get("type")
    if not isinstance(kind, str):
        raise malformed("a stream event has no type")
    return kind


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


def _decode_response(
    response: Mapping[str, object], request_id: str | None, provider: str
) -> Decoded:
    items = [json_object(item, "output[]") for item in array(response.get("output"), "output")]
    finish = _finish(response, items)
    parts: list[AssistantPart] = []
    for index, item in enumerate(items):
        parts.extend(_item_parts(item, index, finish, provider))
    usage = openai_platform.usage(response.get("usage"), _USAGE)
    return assemble(parts, finish, usage, request_id)


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
    if isinstance(code, str) and code in _POLICY_CODES:
        return FinishReason.CONTENT_FILTER
    # The failure codes the reference lists, past those the shared table
    # maps, describe the request (invalid images, prompts, residency).
    failure = openai_platform.error_failure(fields, None, unexplained=ErrorKind.BAD_REQUEST)
    raise failure.wire_error("the provider reported the response failed")


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


CODEC = Codec(encode=encode, decode_events=decode_events, classify=openai_platform.classify)
