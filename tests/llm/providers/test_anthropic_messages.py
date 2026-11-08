"""Anthropic Messages: required limits, effort, thinking replay, usage and streams."""

import pytest

from ember.errors import ErrorKind
from ember.llm.provider import Decoded
from ember.llm.providers import anthropic_messages
from ember.llm.providers.wire import WireError
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

WEATHER = ToolSpec("get_weather", "Weather for a city.", {"type": "object"})


def _encode(kit, request: Request, *, catalog_max: int | None = 64000) -> dict:
    spec = kit.spec(
        "anthropic_messages",
        provider="anthropic",
        wire_id="claude-opus-5-5",
        max_output_tokens=catalog_max,
    )
    return anthropic_messages.encode(spec, request, False)


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
    body = _encode(kit, request)
    assert body == {
        "model": "claude-opus-5-5",
        "max_tokens": 64000,
        "messages": [{"role": "user", "content": "hi"}],
        "system": "terse",
        "stop_sequences": ["###"],
        "output_config": {"effort": "xhigh", "format": {"type": "json_schema", "schema": schema}},
    }
    with pytest.raises(ValueError, match="effort"):
        _encode(kit, Request.from_prompt("hi", sampling=Sampling(reasoning_effort="minimal")))


def test_thinking_display_is_an_extension(kit) -> None:
    request = Request(
        (Message.user("hi"),),
        extensions={"thinking": {"type": "adaptive", "display": "summarized"}},
    )
    assert _encode(kit, request)["thinking"] == {"type": "adaptive", "display": "summarized"}


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
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9g="}},
        {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}},
        {"type": "text", "text": "what is this?"},
    ]


def test_tool_round_trip_replays_thinking_blocks_exactly_in_place(kit) -> None:
    response = kit.json("anthropic/message_thinking_tool_use.json")
    decoded = anthropic_messages.decode(response, None)
    assert decoded.message is not None
    request = Request(
        (
            Message.user("Weather in Paris?"),
            decoded.message,
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
    messages = _encode(kit, request)["messages"]
    assert messages[1] == {"role": "assistant", "content": response["content"]}
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


def test_other_apis_reasoning_and_empty_text_are_left_out(kit) -> None:
    turn = Message(
        "assistant",
        (
            ReasoningPart("x", api="openai_responses", state={"type": "reasoning", "id": "rs"}),
            TextPart(""),
            ToolCallPart("toolu_1", "get_weather", {"city": "Oslo"}),
        ),
    )
    request = Request(
        (Message.user("hi"), turn, Message("tool", (ToolResultPart("toolu_1", "ok"),)))
    )
    assert _encode(kit, request)["messages"][1]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}}
    ]


def test_documented_reference_response_usage(kit) -> None:
    decoded = anthropic_messages.decode(kit.json("anthropic/message_reference.json"), "req_9")
    assert decoded.text == "Hi! My name is Claude."
    assert decoded.finish_reason is FinishReason.STOP
    assert decoded.usage == Usage(input=2095, cache_read=2051, cache_write=2051, output=503)
    assert decoded.usage.prompt_tokens == 6197


def test_thinking_tool_use_response(kit) -> None:
    decoded = anthropic_messages.decode(kit.json("anthropic/message_thinking_tool_use.json"), None)
    assert decoded.finish_reason is FinishReason.TOOL_CALLS
    assert decoded.reasoning == "The user wants the weather in Paris; call the tool."
    assert decoded.text == "Let me check."
    assert [(c.id, c.arguments) for c in decoded.tool_calls] == [
        ("toolu_01A09q90qw90lq917835lq9", {"location": "Paris, France"})
    ]
    assert decoded.usage == Usage(input=512, cache_read=1024, output=180, reasoning=120)


def test_documented_mixed_tool_use_ignores_server_tools_and_has_no_usage(kit) -> None:
    decoded = anthropic_messages.decode(kit.json("anthropic/message_mixed_tool_use.json"), None)
    assert [c.name for c in decoded.tool_calls] == ["run_command"]
    assert decoded.finish_reason is FinishReason.TOOL_CALLS
    assert decoded.usage == Usage(measured=False)


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
    body = kit.json("anthropic/message_end_turn.json")
    body["stop_reason"] = stop_reason
    assert anthropic_messages.decode(body, None).finish_reason is finish


def test_unknown_stop_reason_raises(kit) -> None:
    body = kit.json("anthropic/message_end_turn.json")
    body["stop_reason"] = "exception"
    with pytest.raises(WireError, match="exception"):
        anthropic_messages.decode(body, None)


def test_documented_text_stream(kit) -> None:
    items = list(anthropic_messages.decode_events(kit.events("anthropic/stream_text.sse"), "r"))
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


def test_documented_tool_use_stream_without_trailing_blank_line(kit) -> None:
    items = list(
        anthropic_messages.decode_events(kit.events("anthropic/stream_tool_use.sse"), None)
    )
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
    items = list(
        anthropic_messages.decode_events(kit.events("anthropic/stream_thinking.sse"), None)
    )
    reasoning = [item.text for item in items if isinstance(item, ReasoningDelta)]
    assert len(reasoning) == 4
    done = items[-1]
    assert isinstance(done, Decoded) and done.message is not None
    thinking = done.message.parts[0]
    assert thinking == ReasoningPart(
        "".join(reasoning),
        api="anthropic_messages",
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
        for item in anthropic_messages.decode_events(events, None):
            received.append(item)
    assert received == [TextDelta("Hello")]
    assert caught.value.kind is ErrorKind.OVERLOADED
    assert "Overloaded" not in caught.value.reason


def test_unknown_events_are_skipped_and_a_cut_stream_raises(kit) -> None:
    events = kit.events("anthropic/stream_text.sse")
    extra = kit.sse({"type": "future_event", "detail": 1})
    items = list(anthropic_messages.decode_events(events[:2] + extra + events[2:], None))
    assert isinstance(items[-1], Decoded) and items[-1].text == "Hello!"
    with pytest.raises(WireError) as caught:
        list(anthropic_messages.decode_events(events[:-1], None))
    assert caught.value.kind is ErrorKind.TRANSPORT


def test_error_label_reads_the_documented_error_shape(kit) -> None:
    assert anthropic_messages.error_label(kit.json("anthropic/error_not_found.json")) == (
        "not_found_error"
    )
