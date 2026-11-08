"""Gemini: generationConfig, function calls and signatures, finish reasons, streams."""

import pytest

from ember.errors import ErrorKind
from ember.llm.provider import Decoded
from ember.llm.providers import gemini
from ember.llm.providers.wire import WireError
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

WEATHER = ToolSpec("get_weather", "Weather for a city.", {"type": "object"})


def _encode(kit, request: Request) -> dict:
    spec = kit.spec("gemini", provider="gemini", wire_id="gemini-3.8-flash")
    return gemini.encode(spec, request, False)


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


def test_images_are_inline_data_or_file_data(kit) -> None:
    message = Message(
        "user",
        (
            ImagePart("image/png", data=b"\x89PNG"),
            ImagePart("image/jpeg", url="https://example.test/files/abc"),
        ),
    )
    assert _encode(kit, Request((message,)))["contents"][0]["parts"] == [
        {"inlineData": {"mimeType": "image/png", "data": "iVBORw=="}},
        {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://example.test/files/abc"}},
    ]


def test_function_call_round_trip_returns_signatures_on_their_parts(kit) -> None:
    response = kit.json("gemini/function_call.json")
    decoded = gemini.decode(response, None)
    assert decoded.message is not None
    request = Request(
        (
            Message.user("Weather and time in Paris?"),
            decoded.message,
            Message(
                "tool",
                (
                    ToolResultPart("fc-7", "18C"),
                    ToolResultPart("call_1", "clock unavailable", is_error=True),
                ),
            ),
        ),
        tools=(WEATHER,),
    )
    contents = _encode(kit, request)["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            response["candidates"][0]["content"]["parts"][0],
            {
                "functionCall": {"id": "fc-7", "name": "get_weather", "args": {"city": "Paris"}},
                "thoughtSignature": "CiQBVKhc7call",
            },
            {"functionCall": {"id": "call_1", "name": "get_time", "args": {"zone": "CET"}}},
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
            {
                "functionResponse": {
                    "id": "call_1",
                    "name": "get_time",
                    "response": {"error": "clock unavailable"},
                }
            },
        ],
    }


def test_a_trailing_signature_is_returned_on_an_empty_text_part(kit) -> None:
    turn = Message(
        "assistant",
        (
            TextPart("Done."),
            ReasoningPart("", api="gemini", state={"thoughtSignature": "sig-end"}),
            ReasoningPart("", api="anthropic_messages", state={"type": "thinking"}),
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
    decoded = gemini.decode(kit.json("gemini/text.json"), "ignored-header-id")
    assert decoded.text == "AI learns patterns from data."
    assert decoded.finish_reason is FinishReason.STOP
    assert decoded.request_id == "resp-gem-1"
    assert decoded.usage == Usage(input=20, cache_read=100, output=42, reasoning=30)
    assert decoded.message == Message(
        "assistant",
        (
            ReasoningPart("", api="gemini", state={"thoughtSignature": "CiQBVKhc7text"}),
            TextPart("AI learns patterns "),
            TextPart("from data."),
        ),
    )


def test_function_calls_finish_as_tool_calls_and_get_ids(kit) -> None:
    decoded = gemini.decode(kit.json("gemini/function_call.json"), None)
    assert decoded.finish_reason is FinishReason.TOOL_CALLS
    assert [(c.id, c.name) for c in decoded.tool_calls] == [
        ("fc-7", "get_weather"),
        ("call_1", "get_time"),
    ]
    assert decoded.reasoning == "Checking the weather."
    assert decoded.usage == Usage(input=55, output=30, reasoning=10)


def test_blocked_prompt_is_content_filtering(kit) -> None:
    decoded = gemini.decode(kit.json("gemini/prompt_blocked.json"), None)
    assert decoded.finish_reason is FinishReason.CONTENT_FILTER
    assert decoded.message is None and decoded.usage == Usage(input=8)


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
    assert gemini.decode(body, None).finish_reason is finish


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("MALFORMED_FUNCTION_CALL", ErrorKind.SERVER),
        ("MISSING_THOUGHT_SIGNATURE", ErrorKind.BAD_REQUEST),
        ("OTHER", ErrorKind.SERVER),
    ],
)
def test_failed_generations_raise(kit, reason, kind) -> None:
    body = kit.json("gemini/text.json")
    body["candidates"][0]["finishReason"] = reason
    with pytest.raises(WireError) as caught:
        gemini.decode(body, None)
    assert caught.value.kind is kind


def test_missing_usage_is_unmeasured(kit) -> None:
    body = kit.json("gemini/text.json")
    del body["usageMetadata"]
    assert gemini.decode(body, None).usage == Usage(measured=False)


def test_error_body_label(kit) -> None:
    assert (
        gemini.error_label(kit.json("gemini/error_resource_exhausted.json")) == "RESOURCE_EXHAUSTED"
    )


def test_text_stream_merges_chunks_and_keeps_the_signature(kit) -> None:
    items = list(gemini.decode_events(kit.events("gemini/stream_text.sse"), None))
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
            ReasoningPart("", api="gemini", state={"thoughtSignature": "CiQBVKhc7end"}),
            TextPart(" from data."),
        ),
    )


def test_streamed_function_call(kit) -> None:
    chunk = kit.json("gemini/function_call.json")
    items = list(gemini.decode_events(kit.sse(chunk), None))
    calls = [item for item in items if isinstance(item, ToolCallDelta)]
    assert [(c.index, c.id, c.name, c.arguments_json) for c in calls] == [
        (0, "fc-7", "get_weather", '{"city":"Paris"}'),
        (1, "call_1", "get_time", '{"zone":"CET"}'),
    ]
    done = items[-1]
    assert isinstance(done, Decoded) and done.finish_reason is FinishReason.TOOL_CALLS
    assert done.message == gemini.decode(chunk, None).message


def test_stream_without_a_finish_reason_is_cut(kit) -> None:
    events = kit.events("gemini/stream_text.sse")[:2]
    with pytest.raises(WireError) as caught:
        list(gemini.decode_events(events, None))
    assert caught.value.kind is ErrorKind.TRANSPORT


def test_error_chunk_raises_with_its_status(kit) -> None:
    events = kit.events("gemini/stream_text.sse")[:1] + kit.sse(
        kit.json("gemini/error_resource_exhausted.json")
    )
    received = []
    with pytest.raises(WireError) as caught:
        for item in gemini.decode_events(events, None):
            received.append(item)
    assert received == [TextDelta("AI learns")]
    assert caught.value.kind is ErrorKind.RATE_LIMITED
    assert "quota" not in caught.value.reason
