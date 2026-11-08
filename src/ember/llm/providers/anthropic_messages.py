"""Anthropic Messages: the request body and the response, both ways.

This module is the only code that reads or writes Messages shapes. It targets
API version 2026-10-21 (the provider sends that ``anthropic-version``), which
removed ``temperature``, ``top_p`` and ``top_k`` for every model.

Reasoning depth is controlled with ``output_config.effort``; ember never sets
``thinking``. Manual thinking budgets (``budget_tokens``) are deprecated and
rejected by Claude 4.7 and later, so there is no effort-to-budget table. A
caller who wants a ``thinking`` setting, for example ``display``, passes it as
an extension.

Thinking blocks are returned as `ReasoningPart`s whose state is the block
exactly as received, so replaying `Result.message` passes them back
unchanged, as tool use with thinking requires.

Wire reference, read 2026-11-07:
https://docs.claude.com/en/api/messages
https://docs.claude.com/en/docs/build-with-claude/streaming
https://docs.claude.com/en/api/errors
https://docs.claude.com/en/docs/build-with-claude/effort
https://docs.claude.com/en/docs/build-with-claude/extended-thinking
https://docs.claude.com/en/api/versioning
"""

import base64
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
    as_json,
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
    ReasoningPart,
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

__all__ = ["API_VERSION", "CODEC", "decode", "decode_events", "encode", "error_label"]

API_VERSION = "2026-10-21"
"""The ``anthropic-version`` this codec speaks."""

_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})

_STOP_REASONS: Mapping[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    # The provider paused a long turn before it finished; like a token limit,
    # the output is incomplete.
    "pause_turn": FinishReason.LENGTH,
    "refusal": FinishReason.REFUSAL,
}

_ERROR_KINDS: Mapping[str, ErrorKind] = {
    "invalid_request_error": ErrorKind.BAD_REQUEST,
    "request_too_large": ErrorKind.BAD_REQUEST,
    "authentication_error": ErrorKind.AUTH,
    "permission_error": ErrorKind.PERMISSION,
    "billing_error": ErrorKind.PERMISSION,
    "not_found_error": ErrorKind.NOT_FOUND,
    "rate_limit_error": ErrorKind.RATE_LIMITED,
    "overloaded_error": ErrorKind.OVERLOADED,
    "api_error": ErrorKind.SERVER,
    "timeout_error": ErrorKind.SERVER,
}

_REASONING_BLOCKS = frozenset({"thinking", "redacted_thinking"})


def encode(spec: ModelSpec, request: Request, stream: bool) -> Body:
    """Builds the request body.

    Raises:
        ValueError: The request sets something this API version does not
            have (``temperature``, ``top_p``, ``seed``), has no output limit,
            or an extension collides with a field ember sets.
    """
    body: Body = {
        "model": spec.wire_id,
        "max_tokens": _max_tokens(spec, request.sampling),
        "messages": _messages(request.messages),
    }
    if request.system is not None:
        body["system"] = request.system
    if request.sampling.stop:
        body["stop_sequences"] = list(request.sampling.stop)
    output_config = _output_config(request)
    if output_config:
        body["output_config"] = output_config
    if request.tools:
        body["tools"] = [_tool(tool) for tool in request.tools]
        body["tool_choice"] = _tool_choice(request.tool_choice)
    if stream:
        body["stream"] = True
    return merge_extensions(body, request.extensions)


def _max_tokens(spec: ModelSpec, sampling: Sampling) -> int:
    unsupported = {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "seed": sampling.seed,
    }
    for name, value in unsupported.items():
        if value is not None:
            raise ValueError(f"Anthropic API version {API_VERSION} does not accept {name}")
    if sampling.max_output_tokens is not None:
        return sampling.max_output_tokens
    if spec.max_output_tokens is not None:
        return spec.max_output_tokens
    raise ValueError(
        f"Anthropic requires an output limit and the catalog has none for {spec.ref}; "
        "pass max_output_tokens"
    )


def _output_config(request: Request) -> Body:
    config: Body = {}
    effort = request.sampling.reasoning_effort
    if effort is not None:
        if effort not in _EFFORTS:
            raise ValueError(f"Anthropic effort must be one of {sorted(_EFFORTS)}, got {effort!r}")
        config["effort"] = effort
    if request.output_schema is not None:
        config["format"] = {"type": "json_schema", "schema": request.output_schema}
    return config


def _messages(messages: tuple[Message, ...]) -> list[JsonValue]:
    encoded: list[JsonValue] = []
    for message in messages:
        if message.role == "user":
            encoded.append({"role": "user", "content": _user_content(message.parts)})
        elif message.role == "assistant":
            encoded.append({"role": "assistant", "content": list(_assistant_blocks(message))})
        else:
            encoded.append(
                {"role": "user", "content": [_tool_result(part) for part in message.parts]}
            )
    return encoded


def _user_content(parts: tuple[Part, ...]) -> JsonValue:
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    blocks: list[JsonValue] = []
    for part in parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image", "source": _image_source(part)})
    return blocks


def _image_source(part: ImagePart) -> JsonValue:
    if part.url is not None:
        return {"type": "url", "url": part.url}
    assert part.data is not None  # ImagePart holds exactly one of url and data.
    return {
        "type": "base64",
        "media_type": part.media_type,
        "data": base64.b64encode(part.data).decode("ascii"),
    }


def _assistant_blocks(message: Message) -> Iterator[JsonValue]:
    """Encodes an assistant turn's blocks in order.

    Empty text is left out because the API rejects empty text blocks; thinking
    from other wire protocols cannot be read here and is left out.
    """
    for part in message.parts:
        if isinstance(part, TextPart) and part.text:
            yield {"type": "text", "text": part.text}
        elif isinstance(part, ReasoningPart) and part.api == "anthropic_messages":
            yield part.state
        elif isinstance(part, ToolCallPart):
            yield {"type": "tool_use", "id": part.id, "name": part.name, "input": part.arguments}


def _tool_result(part: Part) -> JsonValue:
    assert isinstance(part, ToolResultPart)  # Tool turns hold only tool results.
    block: dict[str, JsonValue] = {
        "type": "tool_result",
        "tool_use_id": part.call_id,
        "content": part.content,
    }
    if part.is_error:
        block["is_error"] = True
    return block


def _tool(tool: ToolSpec) -> JsonValue:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}


def _tool_choice(choice: ToolChoice) -> JsonValue:
    if choice == "required":
        return {"type": "any"}
    if choice in ("auto", "none"):
        return {"type": choice}
    return {"type": "tool", "name": choice}


def decode(body: object, request_id: str | None) -> Decoded:
    """Decodes a complete message.

    Raises:
        WireError: The response is malformed or reports an unknown stop reason.
    """
    message = json_object(body, "message")
    finish = _finish(message.get("stop_reason"))
    parts: list[AssistantPart] = []
    for index, raw in enumerate(array(message.get("content"), "content")):
        part = _content_part(json_object(raw, f"content[{index}]"), index)
        if part is not None:
            parts.append(part)
    return assemble(parts, finish, usage(message.get("usage")), request_id)


def _content_part(block: Mapping[str, object], index: int) -> AssistantPart | None:
    """Maps one content block; server-tool blocks carry no part."""
    kind = block.get("type")
    if kind == "text":
        return TextPart(string(block.get("text"), f"content[{index}].text"))
    if kind == "thinking":
        text = string(block.get("thinking"), f"content[{index}].thinking")
        return ReasoningPart(text=text, api="anthropic_messages", state=as_json(block))
    if kind == "redacted_thinking":
        return ReasoningPart(text="", api="anthropic_messages", state=as_json(block))
    if kind == "tool_use":
        return ToolCallPart(
            id=string(block.get("id"), f"content[{index}].id"),
            name=string(block.get("name"), f"content[{index}].name"),
            arguments=as_json(json_object(block.get("input"), f"content[{index}].input")),
        )
    return None


def _finish(raw: object) -> FinishReason:
    if not isinstance(raw, str) or raw not in _STOP_REASONS:
        raise malformed(f"unknown stop_reason {identifier(raw) or 'missing'}")
    return _STOP_REASONS[raw]


def usage(value: object) -> Usage:
    """Maps ``usage``; ``input_tokens`` excludes cache reads and writes."""
    if value is None:
        return Usage(measured=False)
    fields = json_object(value, "usage")
    details = fields.get("output_tokens_details")
    thinking = json_object(details, "output_tokens_details").get("thinking_tokens", 0)
    output = count(fields.get("output_tokens"), "usage.output_tokens")
    reasoning = count(thinking if details is not None else 0, "usage.thinking_tokens")
    if reasoning > output:
        raise malformed("usage thinking_tokens exceed output_tokens")
    return Usage(
        input=count(fields.get("input_tokens"), "usage.input_tokens"),
        cache_read=_optional_count(fields, "cache_read_input_tokens"),
        cache_write=_optional_count(fields, "cache_creation_input_tokens"),
        output=output,
        reasoning=reasoning,
    )


def _optional_count(fields: Mapping[str, object], key: str) -> int:
    """Reads a count the API documents as ``number or null``; null is zero."""
    value = fields.get(key)
    return 0 if value is None else count(value, f"usage.{key}")


class _MessageStream:
    """State for one streamed message: blocks by index, usage and stop reason."""

    def __init__(self, request_id: str | None) -> None:
        self.request_id = request_id
        self.acc = StreamAccumulator("anthropic_messages")
        self.kinds: dict[int, str] = {}
        self.signatures: dict[int, str] = {}
        self.usage_fields: dict[str, object] = {}
        self.stop_reason: object = None

    def start_message(self, data: Mapping[str, object]) -> None:
        message = json_object(data.get("message"), "message_start.message")
        if message.get("usage") is not None:
            self.usage_fields.update(json_object(message["usage"], "message_start.usage"))

    def start_block(self, data: Mapping[str, object]) -> Iterator[StreamItem]:
        index = count(data.get("index"), "content_block_start.index")
        block = json_object(data.get("content_block"), "content_block_start.content_block")
        kind = string(block.get("type"), "content_block.type")
        self.kinds[index] = kind
        if kind == "text":
            text = string(block.get("text", ""), "content_block.text")
            delta: StreamItem = self.acc.text(index, text)
            if text:
                yield delta
        elif kind == "thinking":
            thinking = string(block.get("thinking", ""), "content_block.thinking")
            self.signatures[index] = string(block.get("signature", ""), "content_block.signature")
            delta = self.acc.reasoning(index, thinking)
            if thinking:
                yield delta
        elif kind == "redacted_thinking":
            self.acc.reasoning_state(index, as_json(block))
        elif kind == "tool_use":
            yield self.acc.tool_call(
                index,
                id=string(block.get("id"), "tool_use.id"),
                name=string(block.get("name"), "tool_use.name"),
                arguments="",
            )

    def delta(self, data: Mapping[str, object]) -> Iterator[StreamItem]:
        index = count(data.get("index"), "content_block_delta.index")
        delta = json_object(data.get("delta"), "content_block_delta.delta")
        kind = delta.get("type")
        block_kind = self.kinds.get(index)
        if kind == "text_delta" and block_kind == "text":
            yield self.acc.text(index, string(delta.get("text"), "text_delta.text"))
        elif kind == "thinking_delta" and block_kind == "thinking":
            thinking = string(delta.get("thinking"), "thinking_delta.thinking")
            if thinking:
                yield self.acc.reasoning(index, thinking)
        elif kind == "signature_delta" and block_kind == "thinking":
            self.signatures[index] = string(delta.get("signature"), "signature_delta")
        elif kind == "input_json_delta" and block_kind == "tool_use":
            partial = string(delta.get("partial_json"), "input_json_delta.partial_json")
            yield self.acc.tool_call(index, id=None, name=None, arguments=partial)

    def stop_block(self, data: Mapping[str, object]) -> None:
        index = count(data.get("index"), "content_block_stop.index")
        if self.kinds.get(index) == "thinking":
            block: dict[str, JsonValue] = {
                "type": "thinking",
                "thinking": self.acc.block_text(index),
                "signature": self.signatures[index],
            }
            self.acc.reasoning_state(index, block)

    def message_delta(self, data: Mapping[str, object]) -> None:
        delta = json_object(data.get("delta"), "message_delta.delta")
        if delta.get("stop_reason") is not None:
            self.stop_reason = delta["stop_reason"]
        if data.get("usage") is not None:
            cumulative = json_object(data["usage"], "message_delta.usage")
            self.usage_fields.update({k: v for k, v in cumulative.items() if v is not None})

    def finish(self) -> Iterator[StreamItem]:
        finish = _finish(self.stop_reason)
        measured = usage(self.usage_fields) if self.usage_fields else Usage(measured=False)
        if measured.measured:
            yield UsageEvent(measured)
        yield assemble(self.acc.parts(finish), finish, measured, self.request_id)


def decode_events(events: Iterable[SseEvent], request_id: str | None) -> Iterator[StreamItem]:
    """Decodes a streamed message.

    Unknown event and delta types are skipped, as the API's versioning policy
    asks; server-tool blocks carry no part.

    Raises:
        WireError: The stream reports an error, is malformed, or ends before
            ``message_stop``.
    """
    state = _MessageStream(request_id)
    for event in events:
        data = parse_event_data(event, "stream event")
        kind = data.get("type")
        if kind == "message_start":
            state.start_message(data)
        elif kind == "content_block_start":
            yield from state.start_block(data)
        elif kind == "content_block_delta":
            yield from state.delta(data)
        elif kind == "content_block_stop":
            state.stop_block(data)
        elif kind == "message_delta":
            state.message_delta(data)
        elif kind == "message_stop":
            yield from state.finish()
            return
        elif kind == "error":
            raise _stream_error(data)
    raise stream_cut("message_stop")


def _stream_error(data: Mapping[str, object]) -> WireError:
    label = error_label(data)
    kind = _ERROR_KINDS.get(label or "", ErrorKind.SERVER)
    reason = "the provider reported an error in the stream"
    return WireError(kind, f"{reason}; provider error {label}" if label else reason)


def error_label(body: object) -> str | None:
    """Returns the ``error.type`` of an Anthropic error body, if short."""
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    return identifier(error.get("type")) if isinstance(error, dict) else None


CODEC = Codec(encode=encode, decode=decode, decode_events=decode_events)
