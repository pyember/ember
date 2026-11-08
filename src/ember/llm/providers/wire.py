"""Pieces every codec shares: typed JSON access, errors, and stream assembly.

Codecs are pure functions from ember types to a provider's JSON and back. They
report a problem by raising `WireError`, which carries the kind but not the
provider or model; the provider turns it into a `ModelError` in one place,
adding those fields and the request id. Nothing here inspects a provider's
wire shape; that is each codec's job.
"""

import dataclasses
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Literal, TypeAlias, cast

from ember.errors import ErrorKind, ModelError
from ember.llm.provider import Decoded, StreamItem
from ember.llm.providers.sse import SseEvent
from ember.llm.request import (
    JsonValue,
    Message,
    ReasoningPart,
    Request,
    TextPart,
    ToolCallPart,
    WireApi,
)
from ember.llm.result import (
    FinishReason,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from ember.llm.spec import ModelSpec

__all__ = [
    "AssistantPart",
    "Body",
    "Codec",
    "StreamAccumulator",
    "WireError",
    "array",
    "assemble",
    "as_json",
    "complete_arguments",
    "count",
    "identifier",
    "json_object",
    "malformed",
    "merge_extensions",
    "parse_arguments",
    "parse_event_data",
    "stream_cut",
    "string",
]

Body: TypeAlias = dict[str, JsonValue]
"""A JSON request body as a codec builds it."""

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


class WireError(Exception):
    """A codec found a failure or a malformed response.

    Attributes:
        kind: The error kind the provider reports it as.
        reason: Short explanation written by ember; never upstream text.
        status: HTTP status the provider attached to an error frame, if any.
    """

    def __init__(self, kind: ErrorKind, reason: str, *, status: int | None = None) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.status = status

    def to_model_error(
        self, *, provider: str, model: str, request_id: str | None, before_first_byte: bool
    ) -> ModelError:
        """Returns the `ModelError` this failure is reported as."""
        return ModelError(
            self.kind,
            provider=provider,
            model=model,
            reason=self.reason,
            status=self.status,
            request_id=request_id,
            before_first_byte=before_first_byte,
        )


def malformed(what: str) -> WireError:
    """Returns the error for a response that does not match its documented shape."""
    return WireError(ErrorKind.SERVER, f"malformed response: {what}")


def stream_cut(terminal: str) -> WireError:
    """Returns the error for a stream that ended before its terminal event."""
    return WireError(ErrorKind.TRANSPORT, f"the stream ended before {terminal}")


def identifier(value: object) -> str | None:
    """Returns ``value`` if it is a short code such as ``rate_limit_error``.

    Provider error types and codes are identifiers; anything longer or with
    other characters is free text and is never copied into an ember message.
    """
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
        return value
    return None


def json_object(value: object, what: str) -> Mapping[str, object]:
    """Returns ``value`` as a JSON object, or raises `WireError`."""
    if not isinstance(value, dict):
        raise malformed(f"{what} is not an object")
    return cast(Mapping[str, object], value)


def array(value: object, what: str) -> list[object]:
    """Returns ``value`` as a JSON array, or raises `WireError`."""
    if not isinstance(value, list):
        raise malformed(f"{what} is not an array")
    return cast(list[object], value)


def string(value: object, what: str) -> str:
    """Returns ``value`` as a string, or raises `WireError`."""
    if not isinstance(value, str):
        raise malformed(f"{what} is not a string")
    return value


def count(value: object, what: str) -> int:
    """Returns ``value`` as a token count, or raises `WireError`."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise malformed(f"{what} is not a non-negative integer")
    return value


def parse_event_data(event: SseEvent, what: str) -> Mapping[str, object]:
    """Parses an event's data as a JSON object, or raises `WireError`."""
    try:
        value: object = json.loads(event.data)
    except json.JSONDecodeError:
        raise malformed(f"{what} is not JSON") from None
    return json_object(value, what)


def parse_arguments(raw: str) -> Mapping[str, JsonValue] | None:
    """Parses tool-call arguments; ``None`` when they are not a JSON object."""
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return cast(Mapping[str, JsonValue], value)


def as_json(value: Mapping[str, object]) -> Mapping[str, JsonValue]:
    """Types a mapping that came out of ``json.loads`` as JSON."""
    return cast(Mapping[str, JsonValue], value)


def merge_extensions(body: Body, extensions: Mapping[str, JsonValue]) -> Body:
    """Merges the caller's provider-specific fields into ``body``.

    Objects merge recursively, so ``{"reasoning": {"summary": "auto"}}`` adds
    to the ``reasoning`` object ember built. A field ember already set, at any
    depth, is an error rather than an override.

    Raises:
        ValueError: An extension names a field the codec already sets.
    """
    return _merge(body, extensions, "")


def _merge(
    base: Mapping[str, JsonValue], extra: Mapping[str, JsonValue], path: str
) -> dict[str, JsonValue]:
    merged: dict[str, JsonValue] = dict(base)
    for key, value in extra.items():
        where = f"{path}.{key}" if path else key
        existing = merged.get(key)
        if key not in merged:
            merged[key] = value
        elif isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge(existing, value, where)
        else:
            raise ValueError(f"extension field {where!r} collides with a field ember sets")
    return merged


@dataclasses.dataclass(frozen=True, slots=True)
class Codec:
    """One wire format: request encoding and response decoding.

    Attributes:
        encode: Builds the JSON body for ``(spec, request, stream)``.
        decode: Turns a complete JSON response and the request id into `Decoded`.
        decode_events: Turns a stream's events and the request id into stream
            items ending in exactly one `Decoded`.
    """

    encode: Callable[[ModelSpec, Request, bool], Body]
    decode: Callable[[object, str | None], Decoded]
    decode_events: Callable[[Iterable[SseEvent], str | None], Iterator[StreamItem]]


AssistantPart: TypeAlias = TextPart | ToolCallPart | ReasoningPart


def assemble(
    parts: Sequence[AssistantPart],
    finish_reason: FinishReason,
    usage: Usage,
    request_id: str | None,
) -> Decoded:
    """Builds `Decoded` from the assistant parts in the order the provider sent them.

    `Decoded.text`, `reasoning` and `tool_calls` are derived from the parts, so
    they always agree with the replayable `Decoded.message`.
    """
    return Decoded(
        text="".join(part.text for part in parts if isinstance(part, TextPart)),
        reasoning="".join(part.text for part in parts if isinstance(part, ReasoningPart)),
        tool_calls=tuple(
            ToolCall(part.id, part.name, part.arguments)
            for part in parts
            if isinstance(part, ToolCallPart)
        ),
        finish_reason=finish_reason,
        usage=usage,
        request_id=request_id,
        message=Message("assistant", tuple(parts)) if parts else None,
    )


@dataclasses.dataclass(slots=True)
class _Block:
    kind: Literal["text", "reasoning", "tool"]
    pieces: list[str] = dataclasses.field(default_factory=list)
    call_index: int = -1
    call_id: str | None = None
    call_name: str | None = None
    state: Mapping[str, JsonValue] = dataclasses.field(default_factory=dict)


class StreamAccumulator:
    """Assembles a stream's pieces into ordered assistant parts.

    A provider streams content in blocks (an Anthropic content block, a
    Gemini part, a Chat tool call). Each piece names its block with a key the
    codec chooses; blocks become parts in the order they first appeared. Each
    method records a piece and returns the item to yield, so a codec's stream
    loop reads ``yield acc.text(key, delta)``.
    """

    def __init__(self, api: WireApi) -> None:
        self._api: WireApi = api
        self._blocks: dict[object, _Block] = {}
        self._calls = 0

    def text(self, key: object, piece: str) -> TextDelta:
        """Records generated text in block ``key``."""
        self._block(key, "text").pieces.append(piece)
        return TextDelta(piece)

    def reasoning(self, key: object, piece: str) -> ReasoningDelta:
        """Records reasoning summary text in block ``key``."""
        self._block(key, "reasoning").pieces.append(piece)
        return ReasoningDelta(piece)

    def reasoning_state(self, key: object, state: Mapping[str, JsonValue]) -> None:
        """Sets the opaque state replayed with reasoning block ``key``."""
        self._block(key, "reasoning").state = state

    def block_text(self, key: object) -> str:
        """Returns the text received so far in block ``key``."""
        return "".join(self._blocks[key].pieces)

    def tool_call(
        self, key: object, *, id: str | None, name: str | None, arguments: str
    ) -> ToolCallDelta:
        """Records a piece of the tool call in block ``key``.

        Raises:
            WireError: The stream gave one call two different ids or names.
        """
        block = self._block(key, "tool")
        if id is not None:
            if block.call_id not in (None, id):
                raise malformed(f"tool call {block.call_index} changed its id")
            block.call_id = id
        if name is not None:
            if block.call_name not in (None, name):
                raise malformed(f"tool call {block.call_index} changed its name")
            block.call_name = name
        block.pieces.append(arguments)
        return ToolCallDelta(index=block.call_index, id=id, name=name, arguments_json=arguments)

    @property
    def has_tool_calls(self) -> bool:
        """Whether any tool call has started."""
        return self._calls > 0

    def parts(self, finish_reason: FinishReason) -> list[AssistantPart]:
        """Returns the assembled parts in stream order.

        Raises:
            WireError: A tool call never received its id or name, or its
                arguments are not a JSON object and the output was not
                truncated.
        """
        parts: list[AssistantPart] = []
        for block in self._blocks.values():
            text = "".join(block.pieces)
            if block.kind == "text":
                parts.append(TextPart(text))
            elif block.kind == "reasoning":
                parts.append(ReasoningPart(text=text, api=self._api, state=block.state))
            else:
                call = _complete_call(block, text, finish_reason)
                if call is not None:
                    parts.append(call)
        return parts

    def _block(self, key: object, kind: Literal["text", "reasoning", "tool"]) -> _Block:
        block = self._blocks.get(key)
        if block is None:
            block = self._blocks[key] = _Block(kind)
            if kind == "tool":
                block.call_index = self._calls
                self._calls += 1
        elif block.kind != kind:
            raise malformed(f"stream block {key!r} changed from {block.kind} to {kind}")
        return block


def _complete_call(block: _Block, raw: str, finish_reason: FinishReason) -> ToolCallPart | None:
    if block.call_id is None or block.call_name is None:
        raise malformed(f"tool call {block.call_index} has no id or name")
    arguments = complete_arguments(raw, finish_reason, block.call_index)
    if arguments is None:
        return None
    return ToolCallPart(block.call_id, block.call_name, arguments)


def complete_arguments(
    raw: str, finish_reason: FinishReason, index: int
) -> Mapping[str, JsonValue] | None:
    """Parses a tool call's arguments, allowing only truncation to cut them.

    Returns:
        The arguments, or ``None`` when generation hit the length limit inside
        this call; the call is then incomplete, and `FinishReason.LENGTH` on
        the result is what tells the caller.

    Raises:
        WireError: The arguments are not a JSON object and the output was
            not truncated.
    """
    arguments = parse_arguments(raw)
    if arguments is None and finish_reason is not FinishReason.LENGTH:
        raise malformed(f"tool call {index} arguments are not a JSON object")
    return arguments
