"""Gemini: generationConfig, function calls and signatures, finish reasons, streams.

Every call streams. A ``GenerateContentResponse`` is also one stream chunk,
so the response fixtures are decoded as single-chunk streams.
"""

import copy

import pytest

from ember.errors import ErrorKind
from ember.llm.provider import Decoded
from ember.llm.providers import gemini
from ember.llm.providers.wire import WireError
from ember.llm.ref import ModelRef
from ember.llm.request import (
    ImagePart,
    Message,
    ReasoningPart,
    Request,
    Sampling,
    TextPart,
    ToolName,
    ToolResultPart,
    ToolSpec,
)
from ember.llm.result import FinishReason, TextDelta, ToolCallDelta, Usage, UsageEvent
from ember.llm.spec import ModelSpec

WEATHER = ToolSpec("get_weather", "Weather for a city.", {"type": "object"})
SPEC = ModelSpec(
    ref=ModelRef("gemini", "gemini-3.8-flash"), wire_id="gemini-3.8-flash", api="gemini"
)
FILES_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc-123"
ASK = Request.from_prompt("hi")


def _encode(kit, request: Request, spec: ModelSpec = SPEC) -> dict:
    return gemini.encode(spec, request)


def _final(kit, *chunks: dict) -> Decoded:
    final = list(gemini.decode_events(SPEC, ASK, kit.sse(*chunks), "ignored-header-id"))[-1]
    assert isinstance(final, Decoded)
    return final


def test_generation_config_uses_documented_names(kit) -> None:
    sampling = Sampling(
        temperature=1.0,
        top_p=0.8,
        max_output_tokens=800,
        stop=("Title",),
        seed=11,
        reasoning_effort="low",
    )
    schema = {"type": "object"}
    request = Request(
        (Message.user("Explain AI"),), system="terse", sampling=sampling, output_schema=schema
    )
    assert _encode(kit, request) == {
        "contents": [{"role": "user", "parts": [{"text": "Explain AI"}]}],
        "systemInstruction": {"parts": [{"text": "terse"}]},
        "generationConfig": {
            "temperature": 1.0,
            "topP": 0.8,
            "maxOutputTokens": 800,
            "stopSequences": ["Title"],
            "seed": 11,
            "thinkingConfig": {"thinkingLevel": "LOW"},
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }


def test_effort_without_a_thinking_level_is_rejected(kit) -> None:
    with pytest.raises(ValueError, match="effort"):
        _encode(kit, Request.from_prompt("hi", sampling=Sampling(reasoning_effort="xhigh")))


def test_extension_adds_to_generation_config(kit) -> None:
    request = Request(
        (Message.user("hi"),),
        sampling=Sampling(reasoning_effort="high"),
        extensions={"generationConfig": {"thinkingConfig": {"includeThoughts": True}}},
    )
    config = _encode(kit, request)["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "HIGH", "includeThoughts": True}


@pytest.mark.parametrize(
    ("choice", "config"),
    [
        ("auto", {"mode": "AUTO"}),
        ("none", {"mode": "NONE"}),
        ("required", {"mode": "ANY"}),
        (ToolName("get_weather"), {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}),
    ],
)
def test_tools_and_tool_choice(kit, choice, config) -> None:
    body = _encode(kit, Request((Message.user("hi"),), tools=(WEATHER,), tool_choice=choice))
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Weather for a city.",
                    "parametersJsonSchema": {"type": "object"},
                }
            ]
        }
    ]
    assert body["toolConfig"] == {"functionCallingConfig": config}


def test_images_are_inline_data_or_files_api_uris(kit) -> None:
    message = Message(
        "user",
        (ImagePart("image/png", data=b"\x89PNG"), ImagePart("image/jpeg", url=FILES_URI)),
    )
    assert _encode(kit, Request((message,)))["contents"][0]["parts"] == [
        {"inlineData": {"mimeType": "image/png", "data": "iVBORw=="}},
        {"fileData": {"mimeType": "image/jpeg", "fileUri": FILES_URI}},
    ]


def test_other_image_urls_are_rejected(kit) -> None:
    message = Message("user", (ImagePart("image/png", url="https://example.test/a.png"),))
    with pytest.raises(ValueError, match="Files API"):
        _encode(kit, Request((message,)))


def _id_less_call(first: Decoded) -> str:
    call_id = first.tool_calls[1].id
    assert call_id.startswith("ember-gemini-")
    return call_id


def _loop_request(first: Decoded) -> Request:
    return Request(
        (
            Message.user("Weather and time in Paris?"),
            first.message,
            Message(
                "tool",
                (
                    ToolResultPart("fc-7", "18C"),
                    ToolResultPart(_id_less_call(first), "clock unavailable", is_error=True),
                ),
            ),
        ),
        tools=(WEATHER,),
    )


def test_two_turn_tool_loop_returns_signatures_on_their_parts(kit) -> None:
    response = kit.json("gemini/function_call.json")
    first = _final(kit, response)
    contents = _encode(kit, _loop_request(first))["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            response["candidates"][0]["content"]["parts"][0],
            {
                "functionCall": {"id": "fc-7", "name": "get_weather", "args": {"city": "Paris"}},
                "thoughtSignature": "CiQBVKhc7call",
            },
            {"functionCall": {"name": "get_time", "args": {"zone": "CET"}}},
        ],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "id": "fc-7",
                    "name": "get_weather",
                    "response": {"output": "18C"},
                }
            },
            {"functionResponse": {"name": "get_time", "response": {"error": "clock unavailable"}}},
        ],
    }


def test_id_less_calls_get_ids_unique_across_turns(kit) -> None:
    turn_one = kit.json("gemini/function_call.json")
    turn_two = copy.deepcopy(turn_one)
    turn_two["responseId"] = "resp-gem-later"
    first, second = _final(kit, turn_one), _final(kit, turn_two)
    assert _id_less_call(first) != _id_less_call(second)
    assert _id_less_call(first) == _id_less_call(_final(kit, turn_one))


def test_signatures_and_thoughts_from_another_provider_are_not_sent(kit) -> None:
    first = _final(kit, kit.json("gemini/function_call.json"))
    vertex = ModelSpec(
        ref=ModelRef("vertex", "gemini-3.8-flash"), wire_id="gemini-3.8-flash", api="gemini"
    )
    assert _encode(kit, _loop_request(first), vertex)["contents"][1] == {
        "role": "model",
        "parts": [
            {"functionCall": {"id": "fc-7", "name": "get_weather", "args": {"city": "Paris"}}},
            {"functionCall": {"name": "get_time", "args": {"zone": "CET"}}},
        ],
    }


def test_a_trailing_signature_is_returned_on_an_empty_text_part(kit) -> None:
    turn = Message(
        "assistant",
        (
            TextPart("Done."),
            ReasoningPart(
                "", api="gemini", provider="gemini", state={"thoughtSignature": "sig-end"}
            ),
            ReasoningPart(
                "", api="anthropic_messages", provider="anthropic", state={"type": "thinking"}
            ),
        ),
    )
    contents = _encode(kit, Request((Message.user("hi"), turn, Message.user("ok"))))["contents"]
    assert contents[1]["parts"] == [{"text": "Done."}, {"text": "", "thoughtSignature": "sig-end"}]


def test_tool_result_for_an_unknown_call_is_rejected(kit) -> None:
    request = Request(
        (
            Message.user("hi"),
            Message("assistant", (TextPart("x"),)),
            Message("tool", (ToolResultPart("nope", "1"),)),
        )
    )
    with pytest.raises(ValueError, match="nope"):
        _encode(kit, request)


def test_text_response_usage_and_request_id(kit) -> None:
    decoded = _final(kit, kit.json("gemini/text.json"))
    assert decoded.text == "AI learns patterns from data."
    assert decoded.finish_reason is FinishReason.STOP
    assert decoded.request_id == "resp-gem-1"
    assert decoded.usage == Usage(input=20, cache_read=100, output=42, reasoning=30)
    assert decoded.message == Message(
        "assistant",
        (
            ReasoningPart(
                "", api="gemini", provider="gemini", state={"thoughtSignature": "CiQBVKhc7text"}
            ),
            # A stream splits one text part across chunks, so adjacent text
            # fragments join; the signature stays before the text it came on.
            TextPart("AI learns patterns from data."),
        ),
    )


def test_function_calls_finish_as_tool_calls(kit) -> None:
    decoded = _final(kit, kit.json("gemini/function_call.json"))
    assert decoded.finish_reason is FinishReason.TOOL_CALLS
    assert [c.name for c in decoded.tool_calls] == ["get_weather", "get_time"]
    assert decoded.tool_calls[0].id == "fc-7"
    assert decoded.reasoning == "Checking the weather."
    assert decoded.usage == Usage(input=55, output=30, reasoning=10)


def test_an_empty_thought_part_keeps_its_signature(kit) -> None:
    chunk = kit.json("gemini/text.json")
    parts = chunk["candidates"][0]["content"]["parts"]
    parts.insert(0, {"text": "", "thought": True, "thoughtSignature": "sig-thought"})
    parts.insert(1, {"text": "", "thought": True})
    decoded = _final(kit, chunk)
    assert decoded.message.parts[0] == ReasoningPart(
        "",
        api="gemini",
        provider="gemini",
        state={"text": "", "thought": True, "thoughtSignature": "sig-thought"},
    )


def test_blocked_prompt_is_content_filtering(kit) -> None:
    decoded = _final(kit, kit.json("gemini/prompt_blocked.json"))
    assert decoded.finish_reason is FinishReason.CONTENT_FILTER
    assert decoded.message == Message("assistant", (TextPart(""),))
    assert decoded.usage == Usage(input=8)


@pytest.mark.parametrize(
    ("reason", "finish"),
    [
        ("MAX_TOKENS", FinishReason.LENGTH),
        ("CONTINUATION", FinishReason.LENGTH),
        ("SAFETY", FinishReason.CONTENT_FILTER),
        ("RECITATION", FinishReason.CONTENT_FILTER),
        ("PROHIBITED_CONTENT", FinishReason.CONTENT_FILTER),
        ("SPII", FinishReason.CONTENT_FILTER),
    ],
)
def test_finish_reasons_are_never_reported_as_stop(kit, reason, finish) -> None:
    body = kit.json("gemini/text.json")
    body["candidates"][0]["finishReason"] = reason
    assert _final(kit, body).finish_reason is finish


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("MALFORMED_FUNCTION_CALL", ErrorKind.SERVER),
        ("MISSING_THOUGHT_SIGNATURE", ErrorKind.BAD_REQUEST),
        ("OTHER", ErrorKind.SERVER),
        ("SOMETHING_NEW", ErrorKind.MALFORMED),
    ],
)
def test_failed_or_unknown_finishes_raise(kit, reason, kind) -> None:
    body = kit.json("gemini/text.json")
    body["candidates"][0]["finishReason"] = reason
    with pytest.raises(WireError) as caught:
        _final(kit, body)
    assert caught.value.kind is kind


def test_missing_usage_is_unmeasured(kit) -> None:
    body = kit.json("gemini/text.json")
    del body["usageMetadata"]
    assert _final(kit, body).usage == Usage(measured=False)


def test_http_errors_carry_status_name_and_retry_delay(kit) -> None:
    body = kit.json("gemini/error_resource_exhausted.json")
    body["error"]["details"] = [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}
    ]
    failure = gemini.classify(429, body)
    assert (failure.kind, failure.label, failure.retry_after_s) == (
        ErrorKind.RATE_LIMITED,
        "RESOURCE_EXHAUSTED",
        27.0,
    )


def test_text_stream_merges_chunks_and_keeps_the_signature(kit) -> None:
    items = list(gemini.decode_events(SPEC, ASK, kit.events("gemini/stream_text.sse"), None))
    usage = Usage(input=9, output=10, reasoning=4)
    assert items[:-1] == [
        TextDelta("AI learns"),
        TextDelta(" patterns"),
        TextDelta(" from data."),
        UsageEvent(usage),
    ]
    done = items[-1]
    assert isinstance(done, Decoded)
    assert (done.text, done.finish_reason, done.usage, done.request_id) == (
        "AI learns patterns from data.",
        FinishReason.STOP,
        usage,
        "resp-gem-s1",
    )
    assert done.message == Message(
        "assistant",
        (
            TextPart("AI learns patterns"),
            ReasoningPart(
                "", api="gemini", provider="gemini", state={"thoughtSignature": "CiQBVKhc7end"}
            ),
            TextPart(" from data."),
        ),
    )


def test_streamed_function_call_deltas(kit) -> None:
    items = list(
        gemini.decode_events(SPEC, ASK, kit.sse(kit.json("gemini/function_call.json")), None)
    )
    calls = [item for item in items if isinstance(item, ToolCallDelta)]
    assert [(c.index, c.name, c.arguments_json) for c in calls] == [
        (0, "get_weather", '{"city":"Paris"}'),
        (1, "get_time", '{"zone":"CET"}'),
    ]


def test_stream_without_a_finish_reason_is_cut(kit) -> None:
    events = kit.events("gemini/stream_text.sse")[:2]
    with pytest.raises(WireError) as caught:
        list(gemini.decode_events(SPEC, ASK, events, None))
    assert caught.value.kind is ErrorKind.TRANSPORT


def test_error_chunk_raises_with_its_status(kit) -> None:
    events = kit.events("gemini/stream_text.sse")[:1] + kit.sse(
        kit.json("gemini/error_resource_exhausted.json")
    )
    received = []
    with pytest.raises(WireError) as caught:
        for item in gemini.decode_events(SPEC, ASK, events, None):
            received.append(item)
    assert received == [TextDelta("AI learns")]
    assert caught.value.kind is ErrorKind.RATE_LIMITED
    assert "quota" not in caught.value.reason


def test_identical_id_less_calls_in_different_turns_get_different_ids(kit) -> None:
    chunk = kit.json("gemini/function_call.json")
    del chunk["responseId"]
    later = Request(
        (Message.user("a"), Message.assistant("b"), Message.user("c")),
    )
    events = kit.sse(chunk)
    first = list(gemini.decode_events(SPEC, ASK, events, None))[-1]
    again = list(gemini.decode_events(SPEC, ASK, kit.sse(chunk), None))[-1]
    second = list(gemini.decode_events(SPEC, later, kit.sse(chunk), None))[-1]
    assert _id_less_call(first) == _id_less_call(again)
    assert _id_less_call(first) != _id_less_call(second)
