"""Anthropic Messages: required limits, effort, thinking replay, usage and streams.

Every call streams, so responses are decoded from event streams: the
documented ones in the fixtures, assembled ones built from documented
message content, or those with the one documented field a test is about
changed.
"""

import json

import pytest

from ember.errors import ErrorKind
from ember.llm.provider import Decoded
from ember.llm.providers import anthropic_messages
from ember.llm.providers.sse import SseEvent
from ember.llm.providers.wire import WireError
from ember.llm.ref import ModelRef
from ember.llm.request import (
    ImagePart,
    Message,
    ReasoningPart,
    Request,
    Sampling,
    TextPart,
    ToolCallPart,
    ToolName,
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

WEATHER = ToolSpec("get_weather", "Weather for a city.", {"type": "object"})
SPEC = ModelSpec(
    ref=ModelRef("anthropic", "claude-opus-5-5"),
    wire_id="claude-opus-5-5",
    api="anthropic_messages",
    max_output_tokens=64000,
)
ASK = Request.from_prompt("hi")


def _encode(kit, request: Request, *, catalog_max: int | None = 64000) -> dict:
    spec = kit.spec(
        "anthropic_messages",
        provider="anthropic",
        wire_id="claude-opus-5-5",
        max_output_tokens=catalog_max,
    )
    return anthropic_messages.encode(spec, request)


def _decode(events: list[SseEvent], request_id: str | None = None) -> list:
    return list(anthropic_messages.decode_events(SPEC, ASK, events, request_id))


def _final(events: list[SseEvent]) -> Decoded:
    final = _decode(events)[-1]
    assert isinstance(final, Decoded)
    return final


def _edited(kit, name: str, event_type: str, edit) -> list[SseEvent]:
    """The fixture's events with ``edit`` applied to the data of each ``event_type`` event."""
    events = []
    for event in kit.events(name):
        data = json.loads(event.data)
        if data["type"] == event_type:
            edit(data)
        events.append(SseEvent(event.event, json.dumps(data)))
    return events


def test_max_tokens_comes_from_the_request_then_the_catalog_and_is_never_invented(kit) -> None:
    capped = Request.from_prompt("hi", sampling=Sampling(max_output_tokens=300))
    assert _encode(kit, capped)["max_tokens"] == 300
    assert _encode(kit, Request.from_prompt("hi"))["max_tokens"] == 64000
    with pytest.raises(ValueError, match="max_output_tokens"):
        _encode(kit, Request.from_prompt("hi"), catalog_max=None)


@pytest.mark.parametrize(
    "sampling", [Sampling(temperature=0.5), Sampling(top_p=0.9), Sampling(seed=3)]
)
def test_settings_removed_in_api_version_2026_10_21_are_rejected(kit, sampling) -> None:
    with pytest.raises(ValueError, match="2026-10-21"):
        _encode(kit, Request.from_prompt("hi", sampling=sampling))


def test_effort_and_schema_go_in_output_config_and_thinking_is_never_set(kit) -> None:
    schema = {"type": "object"}
    request = Request(
        (Message.user("hi"),),
        system="terse",
        sampling=Sampling(reasoning_effort="xhigh", stop=("###",)),
        output_schema=schema,
    )
    assert _encode(kit, request) == {
        "model": "claude-opus-5-5",
        "max_tokens": 64000,
        "messages": [{"role": "user", "content": "hi"}],
        "system": "terse",
        "stop_sequences": ["###"],
        "output_config": {"effort": "xhigh", "format": {"type": "json_schema", "schema": schema}},
        "stream": True,
    }
    with pytest.raises(ValueError, match="effort"):
        _encode(kit, Request.from_prompt("hi", sampling=Sampling(reasoning_effort="minimal")))


def test_thinking_display_is_an_extension(kit) -> None:
    thinking = {"type": "adaptive", "display": "summarized"}
    request = Request((Message.user("hi"),), extensions={"thinking": thinking})
    assert _encode(kit, request)["thinking"] == thinking


@pytest.mark.parametrize(
    ("choice", "wire"),
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        (ToolName("get_weather"), {"type": "tool", "name": "get_weather"}),
    ],
)
def test_tool_choice(kit, choice, wire) -> None:
    body = _encode(kit, Request((Message.user("hi"),), tools=(WEATHER,), tool_choice=choice))
    assert body["tool_choice"] == wire
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Weather for a city.",
            "input_schema": {"type": "object"},
        }
    ]


def test_images_use_base64_or_url_sources(kit) -> None:
    message = Message(
        "user",
        (
            ImagePart("image/jpeg", data=b"\xff\xd8"),
            ImagePart("image/png", url="https://example.test/a.png"),
            TextPart("what is this?"),
        ),
    )
    assert _encode(kit, Request((message,)))["messages"][0]["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9g="},
        },
        {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}},
        {"type": "text", "text": "what is this?"},
    ]


def _loop_request(first: Decoded) -> Request:
    return Request(
        (
            Message.user("Weather in Paris?"),
            first.message,
            Message(
                "tool",
                (
                    ToolResultPart("toolu_01A09q90qw90lq917835lq9", "18C"),
                    ToolResultPart("toolu_missing", "timeout", is_error=True),
                ),
            ),
        ),
        tools=(WEATHER,),
    )


def test_two_turn_tool_loop_replays_thinking_blocks_exactly_in_place(kit) -> None:
    first = _final(kit.events("anthropic/stream_thinking_tool_use.sse"))
    expected = kit.json("anthropic/message_thinking_tool_use.json")["content"]
    messages = anthropic_messages.encode(SPEC, _loop_request(first))["messages"]
    assert messages[1] == {"role": "assistant", "content": expected}
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                "content": "18C",
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_missing",
                "content": "timeout",
                "is_error": True,
            },
        ],
    }


def test_thinking_from_another_provider_is_not_sent(kit) -> None:
    first = _final(kit.events("anthropic/stream_thinking_tool_use.sse"))
    proxy = ModelSpec(
        ref=ModelRef("proxy", "claude-opus-5-5"),
        wire_id="claude-opus-5-5",
        api="anthropic_messages",
        max_output_tokens=64000,
    )
    expected = kit.json("anthropic/message_thinking_tool_use.json")["content"][2:]
    messages = anthropic_messages.encode(proxy, _loop_request(first))["messages"]
    assert messages[1] == {"role": "assistant", "content": expected}


def test_other_apis_reasoning_and_empty_text_are_left_out(kit) -> None:
    turn = Message(
        "assistant",
        (
            ReasoningPart(
                "x", api="openai_responses", provider="openai", state={"type": "reasoning"}
            ),
            TextPart(""),
            ToolCallPart("toolu_1", "get_weather", {"city": "Oslo"}),
        ),
    )
    tool_turn = Message("tool", (ToolResultPart("toolu_1", "ok"),))
    request = Request((Message.user("hi"), turn, tool_turn))
    assert _encode(kit, request)["messages"][1]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}}
    ]


def test_empty_response_turn_is_left_out_of_the_history(kit) -> None:
    events = [e for e in kit.events("anthropic/stream_text.sse") if "content_block" not in e.event]
    first = _final(events)
    assert first.message == Message("assistant", (TextPart(""),))
    request = Request((Message.user("hi"), first.message, Message.user("still there?")))
    assert anthropic_messages.encode(SPEC, request)["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "still there?"},
    ]


def test_documented_text_stream(kit) -> None:
    items = _decode(kit.events("anthropic/stream_text.sse"), "r")
    usage = Usage(input=25, output=15)
    assert items[:3] == [TextDelta("Hello"), TextDelta("!"), UsageEvent(usage)]
    done = items[3]
    assert isinstance(done, Decoded)
    assert (done.text, done.finish_reason, done.usage, done.request_id) == (
        "Hello!",
        FinishReason.STOP,
        usage,
        "r",
    )


def test_documented_reference_usage_splits_cache_reads_and_writes(kit) -> None:
    reference = kit.json("anthropic/message_reference.json")["usage"]

    def use_reference(data: dict) -> None:
        data["message"]["usage"] = reference

    def keep_output(data: dict) -> None:
        data["usage"] = {"output_tokens": reference["output_tokens"]}

    events = _edited(kit, "anthropic/stream_text.sse", "message_start", use_reference)
    events = [
        SseEvent(e.event, json.dumps(_apply(json.loads(e.data), keep_output, "message_delta")))
        for e in events
    ]
    usage = _final(events).usage
    assert usage == Usage(input=2095, cache_read=2051, cache_write=2051, output=503)
    assert usage.prompt_tokens == 6197


def _apply(data: dict, edit, event_type: str) -> dict:
    if data["type"] == event_type:
        edit(data)
    return data


def test_thinking_tool_use_stream(kit) -> None:
    items = _decode(kit.events("anthropic/stream_thinking_tool_use.sse"))
    done = items[-1]
    assert isinstance(done, Decoded)
    assert done.finish_reason is FinishReason.TOOL_CALLS
    assert done.reasoning == "The user wants the weather in Paris; call the tool."
    assert done.text == "Let me check."
    assert [(c.id, c.arguments) for c in done.tool_calls] == [
        ("toolu_01A09q90qw90lq917835lq9", {"location": "Paris, France"})
    ]
    assert done.usage == Usage(input=512, cache_read=1024, output=180, reasoning=120)


def test_server_tool_blocks_carry_no_part_and_missing_usage_is_unmeasured(kit) -> None:
    done = _final(kit.events("anthropic/stream_mixed_tool_use.sse"))
    assert [c.name for c in done.tool_calls] == ["run_command"]
    assert done.finish_reason is FinishReason.TOOL_CALLS
    assert done.usage == Usage(measured=False)


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("model_context_window_exceeded", FinishReason.LENGTH),
        ("pause_turn", FinishReason.LENGTH),
        ("refusal", FinishReason.REFUSAL),
        ("tool_use", FinishReason.TOOL_CALLS),
    ],
)
def test_every_stop_reason_maps(kit, stop_reason, finish) -> None:
    def set_stop(data: dict) -> None:
        data["delta"]["stop_reason"] = stop_reason

    events = _edited(kit, "anthropic/stream_text.sse", "message_delta", set_stop)
    assert _final(events).finish_reason is finish


def test_unknown_stop_reason_is_malformed(kit) -> None:
    def set_stop(data: dict) -> None:
        data["delta"]["stop_reason"] = "exception"

    events = _edited(kit, "anthropic/stream_text.sse", "message_delta", set_stop)
    with pytest.raises(WireError, match="exception") as caught:
        _decode(events)
    assert caught.value.kind is ErrorKind.MALFORMED


def _zero_argument_call(*deltas: dict) -> list[SseEvent]:
    start = {"type": "tool_use", "id": "toolu_1", "name": "now", "input": {}}
    payloads = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": start},
        *({"type": "content_block_delta", "index": 0, "delta": d} for d in deltas),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    return [SseEvent(p["type"], json.dumps(p)) for p in payloads]


@pytest.mark.parametrize(
    "deltas", [(), ({"type": "input_json_delta", "partial_json": ""},)], ids=["none", "empty"]
)
def test_a_tool_call_without_partial_json_keeps_its_start_input(deltas) -> None:
    items = _decode(_zero_argument_call(*deltas))
    done = items[-1]
    assert isinstance(done, Decoded)
    assert [(c.id, c.name, c.arguments) for c in done.tool_calls] == [("toolu_1", "now", {})]
    calls = [item for item in items if isinstance(item, ToolCallDelta)]
    assert "".join(c.arguments_json for c in calls) == "{}"


def test_a_delta_before_its_block_starts_is_malformed(kit) -> None:
    events = [
        e for e in kit.events("anthropic/stream_text.sse") if e.event != "content_block_start"
    ]
    with pytest.raises(WireError) as caught:
        _decode(events)
    assert caught.value.kind is ErrorKind.MALFORMED


def test_documented_tool_use_stream_without_trailing_blank_line(kit) -> None:
    items = _decode(kit.events("anthropic/stream_tool_use.sse"))
    calls = [item for item in items if isinstance(item, ToolCallDelta)]
    assert (calls[0].index, calls[0].id, calls[0].name) == (
        0,
        "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
        "get_weather",
    )
    done = items[-1]
    assert isinstance(done, Decoded)
    assert done.text == "Okay, let's check the weather for San Francisco, CA:"
    assert done.finish_reason is FinishReason.TOOL_CALLS
    assert [c.arguments for c in done.tool_calls] == [{"location": "San Francisco, CA"}]
    assert done.usage == Usage(input=472, output=89)


def test_documented_thinking_stream_keeps_the_signed_block(kit) -> None:
    items = _decode(kit.events("anthropic/stream_thinking.sse"))
    reasoning = [item.text for item in items if isinstance(item, ReasoningDelta)]
    assert len(reasoning) == 4
    done = items[-1]
    assert isinstance(done, Decoded)
    assert done.message.parts[0] == ReasoningPart(
        "".join(reasoning),
        api="anthropic_messages",
        provider="anthropic",
        state={
            "type": "thinking",
            "thinking": "".join(reasoning),
            "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds...",
        },
    )
    assert done.text == "The greatest common divisor of 1071 and 462 is **21**."
    assert done.usage == Usage(measured=False)


def test_documented_error_event_raises_overloaded_and_is_never_text(kit) -> None:
    events = kit.events("anthropic/stream_text.sse")[:4] + kit.events(
        "anthropic/stream_error_overloaded.sse"
    )
    received = []
    with pytest.raises(WireError) as caught:
        for item in anthropic_messages.decode_events(SPEC, ASK, events, None):
            received.append(item)
    assert received == [TextDelta("Hello")]
    assert caught.value.kind is ErrorKind.OVERLOADED
    assert "Overloaded" not in caught.value.reason


def test_unknown_events_are_skipped_and_a_cut_stream_raises(kit) -> None:
    events = kit.events("anthropic/stream_text.sse")
    extra = kit.sse({"type": "future_event", "detail": 1})
    assert _final(events[:2] + extra + events[2:]).text == "Hello!"
    with pytest.raises(WireError) as caught:
        _decode(events[:-1])
    assert caught.value.kind is ErrorKind.TRANSPORT


def test_http_errors_are_classified_by_error_type(kit) -> None:
    not_found = anthropic_messages.classify(404, kit.json("anthropic/error_not_found.json"))
    assert (not_found.kind, not_found.label) == (ErrorKind.NOT_FOUND, "not_found_error")
    billing = {"type": "error", "error": {"type": "billing_error", "message": "card declined"}}
    assert anthropic_messages.classify(402, billing).kind is ErrorKind.QUOTA
    assert anthropic_messages.classify(529, None).kind is ErrorKind.OVERLOADED
