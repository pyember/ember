"""Responses: request fields, terminal statuses, reasoning replay and streams."""

import pytest

from ember.errors import ErrorKind
from ember.llm.provider import Decoded
from ember.llm.providers import openai_responses
from ember.llm.providers.wire import WireError
from ember.llm.request import (
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

WEATHER = ToolSpec("get_current_weather", "Weather for a place.", {"type": "object"})


def _encode(kit, request: Request, stream: bool = False) -> dict:
    spec = kit.spec("openai_responses", wire_id="gpt-6-astra")
    return openai_responses.encode(spec, request, stream)


def test_fields_follow_the_responses_reference(kit) -> None:
    schema = {"type": "object"}
    sampling = Sampling(temperature=0.5, max_output_tokens=900, reasoning_effort="low")
    request = Request(
        (Message.user("hi"),),
        system="be brief",
        sampling=sampling,
        output_schema=schema,
        tools=(WEATHER,),
        tool_choice=ToolName("get_current_weather"),
    )
    assert _encode(kit, request, stream=True) == {
        "model": "gpt-6-astra",
        "input": [{"role": "user", "content": "hi"}],
        "instructions": "be brief",
        "temperature": 0.5,
        "max_output_tokens": 900,
        "reasoning": {"effort": "low"},
        "tools": [
            {
                "type": "function",
                "name": "get_current_weather",
                "description": "Weather for a place.",
                "parameters": schema,
                "strict": False,
            }
        ],
        "tool_choice": {"type": "function", "name": "get_current_weather"},
        "text": {
            "format": {"type": "json_schema", "name": "output", "schema": schema, "strict": True}
        },
        "stream": True,
    }


@pytest.mark.parametrize("sampling", [Sampling(stop=("x",)), Sampling(seed=1)])
def test_settings_the_api_lacks_are_rejected(kit, sampling: Sampling) -> None:
    with pytest.raises(ValueError, match="Responses API has no"):
        _encode(kit, Request.from_prompt("hi", sampling=sampling))


def test_extension_adds_a_reasoning_summary_next_to_the_effort(kit) -> None:
    request = Request(
        (Message.user("hi"),),
        sampling=Sampling(reasoning_effort="high"),
        extensions={"reasoning": {"summary": "auto"}},
    )
    assert _encode(kit, request)["reasoning"] == {"effort": "high", "summary": "auto"}


def test_history_becomes_input_items_and_replays_only_this_apis_reasoning(kit) -> None:
    item = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA"}
    request = Request(
        (
            Message.user("Weather?"),
            Message(
                "assistant",
                (
                    ReasoningPart("", api="openai_responses", state=item),
                    ReasoningPart("x", api="gemini", state={"thoughtSignature": "s"}),
                    TextPart("Checking."),
                    ToolCallPart("call_1", "get_current_weather", {"location": "Boston"}),
                ),
            ),
            Message("tool", (ToolResultPart("call_1", "72F"),)),
        ),
        tools=(WEATHER,),
    )
    assert _encode(kit, request)["input"] == [
        {"role": "user", "content": "Weather?"},
        item,
        {"role": "assistant", "content": "Checking."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_current_weather",
            "arguments": '{"location":"Boston"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "72F"},
    ]


def test_documented_function_call_response(kit) -> None:
    decoded = openai_responses.decode(kit.json("openai/responses_function_call.json"), "req")
    assert decoded.finish_reason is FinishReason.TOOL_CALLS
    assert decoded.message == Message(
        "assistant",
        (
            ToolCallPart(
                "call_unLAR8MvFNptuiZK6K6HCy5k",
                "get_current_weather",
                {"location": "Boston, MA", "unit": "celsius"},
            ),
        ),
    )
    assert decoded.usage == Usage(input=291, output=23)


def test_documented_reasoning_response_splits_reasoning_tokens(kit) -> None:
    decoded = openai_responses.decode(kit.json("openai/responses_reasoning.json"), None)
    assert decoded.text == "The classic tongue twister..."
    assert decoded.finish_reason is FinishReason.STOP
    assert decoded.usage == Usage(input=81, output=1035, reasoning=832)


def test_reasoning_items_become_replayable_parts(kit) -> None:
    body = kit.json("openai/responses_reasoning.json")
    item = {
        "type": "reasoning",
        "id": "rs_9",
        "summary": [{"type": "summary_text", "text": "Thinking about wood."}],
        "encrypted_content": "gAAAAB",
    }
    body["output"].insert(0, item)
    decoded = openai_responses.decode(body, None)
    assert decoded.reasoning == "Thinking about wood."
    assert decoded.message is not None
    assert decoded.message.parts[0] == ReasoningPart(
        "Thinking about wood.", api="openai_responses", state=item
    )


@pytest.mark.parametrize(
    ("reason", "finish"),
    [
        ("max_output_tokens", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("max_messages", FinishReason.LENGTH),
    ],
)
def test_incomplete_responses_are_never_a_clean_stop(kit, reason, finish) -> None:
    body = kit.json("openai/responses_reasoning.json")
    body.update(status="incomplete", incomplete_details={"reason": reason})
    assert openai_responses.decode(body, None).finish_reason is finish


def test_refusal_content_is_a_refusal(kit) -> None:
    body = kit.json("openai/responses_reasoning.json")
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "I can't help."}]
    decoded = openai_responses.decode(body, None)
    assert (decoded.finish_reason, decoded.text) == (FinishReason.REFUSAL, "I can't help.")


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        ("server_error", ErrorKind.SERVER),
        ("rate_limit_exceeded", ErrorKind.RATE_LIMITED),
        ("invalid_prompt", ErrorKind.BAD_REQUEST),
    ],
)
def test_failed_responses_raise_with_the_code_but_not_the_message(kit, code, kind) -> None:
    body = kit.json("openai/responses_reasoning.json")
    body.update(status="failed", error={"code": code, "message": "raw upstream detail"})
    with pytest.raises(WireError) as caught:
        openai_responses.decode(body, None)
    assert caught.value.kind is kind
    assert code in caught.value.reason and "raw upstream" not in caught.value.reason


def test_policy_failure_is_content_filtering(kit) -> None:
    body = kit.json("openai/responses_reasoning.json")
    body.update(status="failed", error={"code": "bio_policy", "message": "blocked"})
    assert openai_responses.decode(body, None).finish_reason is FinishReason.CONTENT_FILTER


def test_cancelled_response_raises_cancelled(kit) -> None:
    body = kit.json("openai/responses_reasoning.json")
    body["status"] = "cancelled"
    with pytest.raises(WireError) as caught:
        openai_responses.decode(body, None)
    assert caught.value.kind is ErrorKind.CANCELLED


def test_documented_text_stream(kit) -> None:
    items = list(
        openai_responses.decode_events(kit.events("openai/responses_stream_text.sse"), "r")
    )
    assert items[0] == TextDelta("Hi")
    assert items[1] == UsageEvent(Usage(input=37, output=11))
    done = items[2]
    assert isinstance(done, Decoded)
    assert done.text == "Hi there! How can I assist you today?"
    assert len(items) == 3


def test_streamed_function_call_and_reasoning(kit) -> None:
    events = kit.events("openai/responses_stream_function_call.sse")
    items = list(openai_responses.decode_events(events, None))
    assert items[0] == ReasoningDelta("Need the weather.")
    calls = [item for item in items if isinstance(item, ToolCallDelta)]
    assert [(c.index, c.id, c.name, c.arguments_json) for c in calls] == [
        (0, "call_unLAR8MvFNptuiZK6K6HCy5k", "get_current_weather", ""),
        (0, None, None, '{"location":'),
        (0, None, None, '"Boston, MA"}'),
    ]
    done = items[-1]
    assert isinstance(done, Decoded)
    assert done.finish_reason is FinishReason.TOOL_CALLS
    assert done.usage == Usage(input=35, cache_read=256, output=23, reasoning=12)
    assert done.message is not None
    first = done.message.parts[0]
    assert isinstance(first, ReasoningPart) and first.state["encrypted_content"] == "gAAAAopaque"


def test_documented_error_event_raises_and_is_never_text(kit) -> None:
    events = kit.events("openai/responses_stream_text.sse")[:5] + kit.sse(
        kit.json("openai/responses_error_event.json")
    )
    received = []
    with pytest.raises(WireError) as caught:
        for item in openai_responses.decode_events(events, None):
            received.append(item)
    assert received == [TextDelta("Hi")]
    assert "Something went wrong" not in caught.value.reason
    assert "ERR_SOMETHING" in caught.value.reason


def test_incomplete_and_failed_terminal_events(kit) -> None:
    prefix = kit.events("openai/responses_stream_text.sse")[:5]
    response = kit.json("openai/responses_reasoning.json")
    incomplete = dict(
        response, status="incomplete", incomplete_details={"reason": "max_output_tokens"}
    )
    items = list(
        openai_responses.decode_events(
            prefix + kit.sse({"type": "response.incomplete", "response": incomplete}), None
        )
    )
    assert isinstance(items[-1], Decoded) and items[-1].finish_reason is FinishReason.LENGTH
    failed = dict(response, status="failed", error={"code": "server_error", "message": "x"})
    with pytest.raises(WireError):
        list(
            openai_responses.decode_events(
                prefix + kit.sse({"type": "response.failed", "response": failed}), None
            )
        )


def test_stream_cut_before_a_terminal_event_is_a_transport_error(kit) -> None:
    events = kit.events("openai/responses_stream_text.sse")[:-1]
    with pytest.raises(WireError) as caught:
        list(openai_responses.decode_events(events, None))
    assert caught.value.kind is ErrorKind.TRANSPORT
