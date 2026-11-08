"""Gemini ``generateContent``: the request body and the response, both ways.

This module is the only code that reads or writes Gemini shapes. A streamed
response (``streamGenerateContent?alt=sse``) is a series of
``GenerateContentResponse`` chunks, the last of which carries the finish
reason.

Reasoning state. Gemini attaches an opaque ``thoughtSignature`` to response
parts and rejects function-call history whose signatures are missing
(finish reason ``MISSING_THOUGHT_SIGNATURE``). A signature becomes a
`ReasoningPart` with state ``{"thoughtSignature": ...}`` placed directly before
the part it was attached to, and encoding attaches it to that part again.
Thought summaries (``thought: true`` parts) become `ReasoningPart`s whose
state is the part as received.

Function-call ids. Gemini may return a ``functionCall`` without an ``id``;
ember then names it ``call_<n>``, its position among the response's calls,
and sends that id back on the call and on its ``functionResponse``. A
``functionResponse`` must also name its function, which encoding looks up
from the earlier call with the same id.

Wire reference, read 2026-11-07 (the reference gives schemas, not example
payloads):
https://ai.google.dev/api/generate-content
https://ai.google.dev/api/generate-content/generate-content
https://ai.google.dev/api/generate-content/stream-generate-content
https://ai.google.dev/gemini-api/docs/troubleshooting
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
    as_json,
    assemble,
    count,
    identifier,
    json_object,
    kind_for_status,
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

__all__ = ["CODEC", "decode", "decode_events", "encode", "error_label"]

#: ``reasoning_effort`` to ``thinkingConfig.thinkingLevel``. Gemini 3 and later
#: take levels; earlier models reject them and are configured by extension.
_THINKING_LEVELS: Mapping[str, str] = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}

_FINISH_REASONS: Mapping[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "CONTINUATION": FinishReason.LENGTH,
    "TOO_MANY_TOOL_CALLS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "LANGUAGE": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
    "IMAGE_SAFETY": FinishReason.CONTENT_FILTER,
    "IMAGE_PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "IMAGE_RECITATION": FinishReason.CONTENT_FILTER,
    "ESCALATION": FinishReason.CONTENT_FILTER,
}

# Finish reasons that mean the call failed rather than ended.
_FAILED_FINISHES: Mapping[str, ErrorKind] = {
    "MISSING_THOUGHT_SIGNATURE": ErrorKind.BAD_REQUEST,
    "UNEXPECTED_TOOL_CALL": ErrorKind.SERVER,
    "MALFORMED_FUNCTION_CALL": ErrorKind.SERVER,
    "MALFORMED_RESPONSE": ErrorKind.SERVER,
    "PUP_LIMITED_DISABLED": ErrorKind.PERMISSION,
    "NO_IMAGE": ErrorKind.SERVER,
    "IMAGE_OTHER": ErrorKind.SERVER,
    "OTHER": ErrorKind.SERVER,
}

_SIGNATURE = "thoughtSignature"


def encode(spec: ModelSpec, request: Request, stream: bool) -> Body:
    """Builds the request body; the model and streaming are in the URL.

    Raises:
        ValueError: A tool result answers no earlier call, the effort has no
            thinking level, or an extension collides with a field ember sets.
    """
    del spec, stream
    body: Body = {"contents": _contents(request.messages)}
    if request.system is not None:
        body["systemInstruction"] = {"parts": [{"text": request.system}]}
    config = _generation_config(request)
    if config:
        body["generationConfig"] = config
    if request.tools:
        body["tools"] = [{"functionDeclarations": [_declaration(t) for t in request.tools]}]
        body["toolConfig"] = {"functionCallingConfig": _calling_config(request.tool_choice)}
    return merge_extensions(body, request.extensions)


def _generation_config(request: Request) -> Body:
    sampling: Sampling = request.sampling
    config: Body = {}
    if sampling.temperature is not None:
        config["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        config["topP"] = sampling.top_p
    if sampling.max_output_tokens is not None:
        config["maxOutputTokens"] = sampling.max_output_tokens
    if sampling.stop:
        config["stopSequences"] = list(sampling.stop)
    if sampling.seed is not None:
        config["seed"] = sampling.seed
    if sampling.reasoning_effort is not None:
        level = _THINKING_LEVELS.get(sampling.reasoning_effort)
        if level is None:
            raise ValueError(
                f"Gemini reasoning effort must be one of {sorted(_THINKING_LEVELS)}, "
                f"got {sampling.reasoning_effort!r}"
            )
        config["thinkingConfig"] = {"thinkingLevel": level}
    if request.output_schema is not None:
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = request.output_schema
    return config


def _contents(messages: tuple[Message, ...]) -> list[JsonValue]:
    contents: list[JsonValue] = []
    call_names: dict[str, str] = {}
    for message in messages:
        if message.role == "user":
            contents.append({"role": "user", "parts": [_user_part(p) for p in message.parts]})
        elif message.role == "assistant":
            call_names.update(
                (part.id, part.name) for part in message.parts if isinstance(part, ToolCallPart)
            )
            contents.append({"role": "model", "parts": _model_parts(message.parts)})
        else:
            parts = [_function_response(part, call_names) for part in message.parts]
            contents.append({"role": "user", "parts": parts})
    return contents


def _user_part(part: Part) -> JsonValue:
    if isinstance(part, TextPart):
        return {"text": part.text}
    assert isinstance(part, ImagePart)  # User turns hold only text and images.
    if part.url is not None:
        return {"fileData": {"mimeType": part.media_type, "fileUri": part.url}}
    assert part.data is not None  # ImagePart holds exactly one of url and data.
    data = base64.b64encode(part.data).decode("ascii")
    return {"inlineData": {"mimeType": part.media_type, "data": data}}


def _model_parts(parts: tuple[Part, ...]) -> list[JsonValue]:
    """Encodes a model turn, putting each signature back on the part it came with."""
    encoded: list[JsonValue] = []
    signature: JsonValue = None
    for part in parts:
        if isinstance(part, ReasoningPart):
            if part.api != "gemini":
                continue
            if set(part.state) == {_SIGNATURE}:
                signature = part.state[_SIGNATURE]
                continue
            encoded.append(part.state)
            continue
        wire: dict[str, JsonValue]
        if isinstance(part, TextPart):
            wire = {"text": part.text}
        else:
            assert isinstance(part, ToolCallPart)  # Model turns hold no other part.
            wire = {"functionCall": {"id": part.id, "name": part.name, "args": part.arguments}}
        if signature is not None:
            wire[_SIGNATURE] = signature
            signature = None
        encoded.append(wire)
    if signature is not None:
        encoded.append({"text": "", _SIGNATURE: signature})
    return encoded


def _function_response(part: Part, call_names: Mapping[str, str]) -> JsonValue:
    assert isinstance(part, ToolResultPart)  # Tool turns hold only tool results.
    name = call_names.get(part.call_id)
    if name is None:
        raise ValueError(
            f"tool result {part.call_id!r} answers no earlier tool call; "
            "Gemini needs the called function's name"
        )
    payload = {"error": part.content} if part.is_error else {"output": part.content}
    return {"functionResponse": {"id": part.call_id, "name": name, "response": payload}}


def _declaration(tool: ToolSpec) -> JsonValue:
    return {
        "name": tool.name,
        "description": tool.description,
        "parametersJsonSchema": tool.parameters,
    }


def _calling_config(choice: ToolChoice) -> JsonValue:
    if choice == "auto":
        return {"mode": "AUTO"}
    if choice == "none":
        return {"mode": "NONE"}
    if choice == "required":
        return {"mode": "ANY"}
    return {"mode": "ANY", "allowedFunctionNames": [choice]}


def decode(body: object, request_id: str | None) -> Decoded:
    """Decodes a complete response.

    Gemini documents no request-id header; the body's ``responseId`` is the
    request id, so ``request_id`` from the transport is not used.

    Raises:
        WireError: The response reports a failure or is malformed.
    """
    del request_id
    response = json_object(body, "response")
    if "error" in response:
        raise stream_error(response["error"])
    response_id = _response_id(response)
    blocked = _blocked_prompt(response)
    if blocked:
        return assemble([], FinishReason.CONTENT_FILTER, _usage_of(response), response_id)
    candidate = _only_candidate(response)
    raw_parts = _candidate_parts(candidate)
    calls = sum(1 for raw in raw_parts if "functionCall" in raw)
    finish = _finish(candidate.get("finishReason"), calls > 0)
    parts: list[AssistantPart] = []
    ordinal = 0
    for raw in raw_parts:
        parts.extend(_parts_of(raw, ordinal))
        if "functionCall" in raw:
            ordinal += 1
    return assemble(parts, finish, _usage_of(response), response_id)


def _response_id(response: Mapping[str, object]) -> str | None:
    value = response.get("responseId")
    return None if value is None else string(value, "responseId")


def _blocked_prompt(response: Mapping[str, object]) -> bool:
    """Whether the prompt was blocked, which Gemini reports with no candidates."""
    feedback = response.get("promptFeedback")
    if feedback is None:
        return False
    return json_object(feedback, "promptFeedback").get("blockReason") is not None


def _only_candidate(response: Mapping[str, object]) -> Mapping[str, object]:
    candidates = array(response.get("candidates"), "candidates")
    if len(candidates) != 1:
        raise malformed(f"expected one candidate, got {len(candidates)}")
    return json_object(candidates[0], "candidates[0]")


def _candidate_parts(candidate: Mapping[str, object]) -> list[Mapping[str, object]]:
    content = candidate.get("content")
    if content is None:
        return []
    raw = json_object(content, "candidate.content").get("parts")
    if raw is None:
        return []
    return [json_object(part, "content.parts[]") for part in array(raw, "content.parts")]


def _parts_of(raw: Mapping[str, object], ordinal: int) -> Iterator[AssistantPart]:
    """Maps one response part; code-execution and media parts carry no part."""
    if raw.get("thought") is True:
        text = string(raw.get("text", ""), "thought text")
        yield ReasoningPart(text=text, api="gemini", state=as_json(raw))
        return
    signature = raw.get(_SIGNATURE)
    if signature is not None:
        state = {_SIGNATURE: string(signature, "thoughtSignature")}
        yield ReasoningPart(text="", api="gemini", state=state)
    if "functionCall" in raw:
        yield _function_call(raw["functionCall"], ordinal)
    elif "text" in raw:
        text = string(raw["text"], "part.text")
        if text:
            yield TextPart(text)


def _function_call(value: object, ordinal: int) -> ToolCallPart:
    call = json_object(value, "functionCall")
    call_id = call.get("id")
    args = call.get("args")
    return ToolCallPart(
        id=f"call_{ordinal}" if call_id is None else string(call_id, "functionCall.id"),
        name=string(call.get("name"), "functionCall.name"),
        arguments={} if args is None else as_json(json_object(args, "functionCall.args")),
    )


def _finish(raw: object, called_tools: bool) -> FinishReason:
    """Maps ``finishReason``; Gemini reports ``STOP`` after a function call."""
    if isinstance(raw, str) and raw in _FAILED_FINISHES:
        reason = f"generation failed with finish reason {raw}"
        raise WireError(_FAILED_FINISHES[raw], reason)
    if not isinstance(raw, str) or raw not in _FINISH_REASONS:
        raise malformed(f"unknown finishReason {identifier(raw) or 'missing'}")
    finish = _FINISH_REASONS[raw]
    if finish is FinishReason.STOP and called_tools:
        return FinishReason.TOOL_CALLS
    return finish


def _usage_of(response: Mapping[str, object]) -> Usage:
    """Maps ``usageMetadata``.

    ``promptTokenCount`` includes cached tokens; tool-use prompt tokens are
    counted separately and are input too. ``candidatesTokenCount`` excludes
    thoughts, so output is candidates plus thoughts.
    """
    value = response.get("usageMetadata")
    if value is None:
        return Usage(measured=False)
    fields = json_object(value, "usageMetadata")
    prompt = count(fields.get("promptTokenCount", 0), "promptTokenCount")
    cached = count(fields.get("cachedContentTokenCount", 0), "cachedContentTokenCount")
    tool_prompt = count(fields.get("toolUsePromptTokenCount", 0), "toolUsePromptTokenCount")
    candidates = count(fields.get("candidatesTokenCount", 0), "candidatesTokenCount")
    thoughts = count(fields.get("thoughtsTokenCount", 0), "thoughtsTokenCount")
    if cached > prompt:
        raise malformed("cachedContentTokenCount exceeds promptTokenCount")
    return Usage(
        input=prompt - cached + tool_prompt,
        cache_read=cached,
        output=candidates + thoughts,
        reasoning=thoughts,
    )


def decode_events(events: Iterable[SseEvent], request_id: str | None) -> Iterator[StreamItem]:
    """Decodes a streamed response.

    Raises:
        WireError: The stream reports an error, is malformed, or ends before
            a chunk with a finish reason.
    """
    del request_id
    stream = _ChunkStream()
    for event in events:
        yield from stream.chunk(parse_event_data(event, "stream chunk"))
    yield from stream.finish()


class _ChunkStream:
    """State for one stream: open block, finish reason, last usage and id."""

    def __init__(self) -> None:
        self.acc = StreamAccumulator("gemini")
        self.block = 0
        self.open_kind = ""
        self.calls = 0
        self.finish_raw: object = None
        self.blocked = False
        self.usage = Usage(measured=False)
        self.response_id: str | None = None

    def chunk(self, chunk: Mapping[str, object]) -> Iterator[StreamItem]:
        if "error" in chunk:
            raise stream_error(chunk["error"])
        self.response_id = self.response_id or _response_id(chunk)
        if chunk.get("usageMetadata") is not None:
            self.usage = _usage_of(chunk)
        if _blocked_prompt(chunk):
            self.blocked = True
            return
        candidates = chunk.get("candidates")
        if candidates is None:
            return
        candidate = _only_candidate(chunk)
        for raw in _candidate_parts(candidate):
            yield from self.part(raw)
        if candidate.get("finishReason") is not None:
            self.finish_raw = candidate["finishReason"]

    def part(self, raw: Mapping[str, object]) -> Iterator[StreamItem]:
        signature = raw.get(_SIGNATURE)
        if raw.get("thought") is True:
            text = string(raw.get("text", ""), "thought text")
            key = self.open_block("thought")
            if text:
                yield self.acc.reasoning(key, text)
            state: dict[str, JsonValue] = {"text": self.acc.block_text(key), "thought": True}
            if signature is not None:
                state[_SIGNATURE] = string(signature, "thoughtSignature")
            self.acc.reasoning_state(key, state)
            return
        if signature is not None:
            sig = {_SIGNATURE: string(signature, "thoughtSignature")}
            self.acc.reasoning_state(self.open_block("signature", fresh=True), sig)
            self.open_kind = ""
        if "functionCall" in raw:
            call = _function_call(raw["functionCall"], self.calls)
            self.calls += 1
            yield self.acc.tool_call(
                self.open_block("call", fresh=True),
                id=call.id,
                name=call.name,
                arguments=json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")),
            )
        elif "text" in raw:
            text = string(raw["text"], "part.text")
            if text:
                yield self.acc.text(self.open_block("text"), text)

    def open_block(self, kind: str, *, fresh: bool = False) -> int:
        """Returns the key for ``kind``, continuing the open block when it matches."""
        if fresh or kind != self.open_kind:
            self.block += 1
            self.open_kind = kind
        return self.block

    def finish(self) -> Iterator[StreamItem]:
        if self.blocked:
            finish = FinishReason.CONTENT_FILTER
        elif self.finish_raw is None:
            raise stream_cut("a chunk with a finishReason")
        else:
            finish = _finish(self.finish_raw, self.calls > 0)
        if self.usage.measured:
            yield UsageEvent(self.usage)
        yield assemble(self.acc.parts(finish), finish, self.usage, self.response_id)


def stream_error(value: object) -> WireError:
    """Returns the error for a Google ``error`` object sent with status 200."""
    fields = json_object(value, "error")
    code = fields.get("code")
    kind = kind_for_status(code) if isinstance(code, int) else ErrorKind.SERVER
    label = identifier(fields.get("status"))
    reason = "the provider reported an error"
    return WireError(kind, f"{reason}; provider error {label}" if label else reason)


def error_label(body: object) -> str | None:
    """Returns the ``error.status`` of a Google error body, if short."""
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    return identifier(error.get("status")) if isinstance(error, dict) else None


CODEC = Codec(encode=encode, decode=decode, decode_events=decode_events)
