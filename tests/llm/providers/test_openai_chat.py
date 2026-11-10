"""Chat Completions: request fields, finish reasons, usage, tools and streams.

Every call streams, so responses are decoded from chunk streams: the
documented ones in the fixtures, or chunks built from them with the one
documented field a test is about changed.
"""

import json

import pytest

from ember.errors import ErrorKind
from ember.llm.provider import Decoded
from ember.llm.providers import openai_chat
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
from ember.llm.result import FinishReason, TextDelta, ToolCallDelta, Usage, UsageEvent
from ember.llm.spec import ModelSpec

WEATHER = ToolSpec("get_weather", "Weather for a city.", {"type": "object"})
SPEC = ModelSpec(ref=ModelRef("openai", "gpt-6-astra"), wire_id="gpt-6-astra", api="openai_chat")
ASK = Request.from_prompt("hi")


def _encode(kit, request: Request) -> dict:
    return openai_chat.encode(SPEC, request)


def _chunks(kit) -> list[dict]:
    """The documented text stream's chunks, without the [DONE] line."""
    return [json.loads(e.data) for e in kit.events("openai/chat_stream_text.sse")[:-1]]


def _decode(kit, chunks: list[dict], done: bool = True) -> list:
    payloads = [*chunks, "[DONE]"] if done else chunks
    return list(openai_chat.decode_events(SPEC, ASK, kit.sse(*payloads), "req"))


def _final(kit, chunks: list[dict]) -> Decoded:
    final = _decode(kit, chunks)[-1]
    assert isinstance(final, Decoded)
    return final


def test_sampling_uses_documented_field_names(kit) -> None:
    sampling = Sampling(
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=256,
        stop=("END",),
        seed=7,
        reasoning_effort="high",
    )
    body = _encode(kit, Request.from_prompt("hi", system="be brief", sampling=sampling))
    assert body == {
        "model": "gpt-6-astra",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_completion_tokens": 256,
        "stop": ["END"],
        "seed": 7,
        "reasoning_effort": "high",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert "max_tokens" not in body


def test_unknown_reasoning_effort_is_rejected_before_sending(kit) -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        _encode(kit, Request.from_prompt("hi", sampling=Sampling(reasoning_effort="extreme")))


def test_streaming_asks_for_usage(kit) -> None:
    body = _encode(kit, Request.from_prompt("hi"))
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    ("choice", "wire"),
    [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        (ToolName("get_weather"), {"type": "function", "function": {"name": "get_weather"}}),
    ],
)
def test_tool_choice(kit, choice, wire) -> None:
    request = Request((Message.user("hi"),), tools=(WEATHER,), tool_choice=choice)
    body = _encode(kit, request)
    assert body["tool_choice"] == wire
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city.",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_tool_history_round_trips_as_assistant_tool_calls_and_tool_messages(kit) -> None:
    request = Request(
        (
            Message.user("Weather in Boston?"),
            Message(
                "assistant",
                (
                    TextPart("Checking."),
                    ReasoningPart(
                        "why",
                        api="anthropic_messages",
                        provider="anthropic",
                        state={"type": "thinking"},
                    ),
                    ToolCallPart("call_1", "get_weather", {"city": "Boston"}),
                ),
            ),
            Message("tool", (ToolResultPart("call_1", "72F"),)),
        ),
        tools=(WEATHER,),
    )
    assert _encode(kit, request)["messages"][1:] == [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Boston"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
    ]


def test_failed_tool_result_has_no_wire_field_and_is_rejected(kit) -> None:
    request = Request(
        (
            Message.user("go"),
            Message("assistant", (ToolCallPart("c", "get_weather", {}),)),
            Message("tool", (ToolResultPart("c", "boom", is_error=True),)),
        )
    )
    with pytest.raises(ValueError, match="is_error"):
        _encode(kit, request)


def test_images_are_urls_or_data_urls(kit) -> None:
    message = Message(
        "user",
        (
            TextPart("compare"),
            ImagePart("image/png", url="https://example.test/a.png"),
            ImagePart("image/png", data=b"\x89PNG"),
        ),
    )
    content = _encode(kit, Request((message,)))["messages"][0]["content"]
    assert [part["image_url"]["url"] for part in content[1:]] == [
        "https://example.test/a.png",
        "data:image/png;base64,iVBORw==",
    ]


def test_output_schema_is_strict_json_schema(kit) -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    body = _encode(kit, Request((Message.user("hi"),), output_schema=schema))
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "schema": schema, "strict": True},
    }


def test_extensions_merge_recursively_and_collisions_raise(kit) -> None:
    extended = Request(
        (Message.user("hi"),),
        sampling=Sampling(max_output_tokens=5),
        extensions={"service_tier": "flex", "stream_options": {"include_obfuscation": False}},
    )
    body = _encode(kit, extended)
    assert body["service_tier"] == "flex"
    assert body["stream_options"] == {"include_usage": True, "include_obfuscation": False}
    for collision in ({"max_completion_tokens": 9}, {"stream_options": {"include_usage": False}}):
        request = Request(
            (Message.user("hi"),), sampling=Sampling(max_output_tokens=5), extensions=collision
        )
        with pytest.raises(ValueError, match="collides"):
            _encode(kit, request)


def test_documented_stream_without_usage_is_unmeasured(kit) -> None:
    items = list(
        openai_chat.decode_events(SPEC, ASK, kit.events("openai/chat_stream_text.sse"), "r")
    )
    assert items[:-1] == [TextDelta("Hello")]
    done = items[-1]
    assert isinstance(done, Decoded)
    assert (done.text, done.finish_reason, done.usage, done.request_id) == (
        "Hello",
        FinishReason.STOP,
        Usage(measured=False),
        "r",
    )
    assert done.message == Message("assistant", (TextPart("Hello"),))


@pytest.mark.parametrize(
    ("finish", "expected"),
    [
        ("length", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("function_call", FinishReason.TOOL_CALLS),
    ],
)
def test_finish_reasons_are_never_reported_as_stop(kit, finish, expected) -> None:
    chunks = _chunks(kit)
    chunks[-1]["choices"][0]["finish_reason"] = finish
    assert _final(kit, chunks).finish_reason is expected


def test_refusal_is_a_refusal_with_its_message_as_text(kit) -> None:
    chunks = _chunks(kit)
    chunks[1]["choices"][0]["delta"] = {"refusal": "I can't help with that."}
    final = _final(kit, chunks)
    assert final.finish_reason is FinishReason.REFUSAL
    assert final.text == "I can't help with that."


def test_unknown_finish_reason_is_malformed(kit) -> None:
    chunks = _chunks(kit)
    chunks[-1]["choices"][0]["finish_reason"] = "abort"
    with pytest.raises(WireError, match="abort") as caught:
        _decode(kit, chunks)
    assert caught.value.kind is ErrorKind.MALFORMED


def test_usage_chunk_splits_cached_cache_written_and_reasoning_tokens(kit) -> None:
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 300,
        "total_tokens": 1300,
        "prompt_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 150},
        "completion_tokens_details": {"reasoning_tokens": 250},
    }
    chunks = [*_chunks(kit), {"id": "chatcmpl-123", "choices": [], "usage": usage}]
    items = _decode(kit, chunks)
    expected = Usage(input=50, cache_read=800, cache_write=150, output=300, reasoning=250)
    assert items[-2] == UsageEvent(expected)
    assert items[-1].usage == expected


def test_null_usage_details_are_zero(kit) -> None:
    usage = {
        "prompt_tokens": 9,
        "completion_tokens": 2,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }
    chunks = [*_chunks(kit), {"id": "chatcmpl-123", "choices": [], "usage": usage}]
    assert _final(kit, chunks).usage == Usage(input=9, output=2)


def test_streamed_tool_calls_assemble_by_index_and_report_usage(kit) -> None:
    events = kit.events("openai/chat_stream_tool_calls.sse")
    items = list(openai_chat.decode_events(SPEC, ASK, events, None))
    deltas = [item for item in items if isinstance(item, ToolCallDelta)]
    assert [(d.index, d.id, d.name) for d in deltas[:2]] == [
        (0, "call_weather", "get_weather"),
        (1, "call_time", "get_time"),
    ]
    usage = Usage(input=20, cache_read=100, output=40, reasoning=16)
    assert items[-2] == UsageEvent(usage)
    done = items[-1]
    assert isinstance(done, Decoded) and done.finish_reason is FinishReason.TOOL_CALLS
    assert [(c.id, c.arguments) for c in done.tool_calls] == [
        ("call_weather", {"city": "Boston"}),
        ("call_time", {"zone": "EST"}),
    ]
    assert done.usage == usage
    assert sum(isinstance(item, Decoded) for item in items) == 1


def _one_call(arguments: str, finish: str) -> list[dict]:
    call = {"index": 0, "id": "call_now", "type": "function"}
    call["function"] = {"name": "now", "arguments": arguments}
    return [
        {"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [call]}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
    ]


def test_a_call_with_empty_arguments_takes_no_arguments(kit) -> None:
    final = _final(kit, _one_call("", "tool_calls"))
    assert [(c.name, c.arguments) for c in final.tool_calls] == [("now", {})]


def test_truncated_tool_call_is_dropped_only_when_length_cut_it(kit) -> None:
    final = _final(kit, _one_call('{"loc', "length"))
    assert final.finish_reason is FinishReason.LENGTH and final.tool_calls == ()
    with pytest.raises(WireError, match="arguments") as caught:
        _decode(kit, _one_call('{"loc', "tool_calls"))
    assert caught.value.kind is ErrorKind.MALFORMED


def test_stream_cut_before_done_is_a_transport_error(kit) -> None:
    with pytest.raises(WireError) as caught:
        _decode(kit, _chunks(kit), done=False)
    assert caught.value.kind is ErrorKind.TRANSPORT


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ({"message": "upstream shard us-east-7 died", "type": "server_error"}, ErrorKind.SERVER),
        ({"message": "out of credit", "code": "insufficient_quota"}, ErrorKind.QUOTA),
    ],
)
def test_error_frame_in_stream_raises_and_is_never_text(kit, error, kind) -> None:
    received = []
    with pytest.raises(WireError) as caught:
        events = kit.sse(*_chunks(kit)[:2], {"error": error})
        for item in openai_chat.decode_events(SPEC, ASK, events, None):
            received.append(item)
    assert received == [TextDelta("Hello")]
    assert caught.value.kind is kind
    assert "shard" not in caught.value.reason and "credit" not in caught.value.reason


def test_two_turn_tool_loop_replays_only_result_message(kit) -> None:
    events = kit.events("openai/chat_stream_tool_calls.sse")
    first = list(openai_chat.decode_events(SPEC, ASK, events, None))[-1]
    assert isinstance(first, Decoded)
    foreign = ReasoningPart("r", api="openai_responses", provider="openai", state={"id": "rs"})
    turn = Message("assistant", (foreign, *first.message.parts))
    request = Request(
        (
            Message.user("Weather in Boston, and the time in New York?"),
            turn,
            Message(
                "tool",
                (ToolResultPart("call_weather", "72F"), ToolResultPart("call_time", "09:00")),
            ),
        ),
        tools=(WEATHER,),
    )
    assert openai_chat.encode(SPEC, request)["messages"] == [
        {"role": "user", "content": "Weather in Boston, and the time in New York?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Boston"}'},
                },
                {
                    "id": "call_time",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": '{"zone":"EST"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": "72F"},
        {"role": "tool", "tool_call_id": "call_time", "content": "09:00"},
    ]
