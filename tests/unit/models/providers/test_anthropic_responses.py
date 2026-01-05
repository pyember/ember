from __future__ import annotations

import json
from types import SimpleNamespace

from ember.models.providers.anthropic_responses import (
    build_anthropic_params,
    stream_events_from_anthropic,
    validate_responses_payload,
)


def _build_request(payload: dict) -> dict:
    payload.setdefault(
        "input",
        [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "ping"}],
            }
        ],
    )
    payload.setdefault("max_output_tokens", 128)
    return payload


def test_build_anthropic_params_maps_tools_and_thinking() -> None:
    payload = _build_request(
        {
            "model": "claude-sonnet-4-5-20250929",
            "instructions": "be helpful",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch weather by city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "parallel_tool_calls": False,
            "reasoning": {"effort": "high"},
        }
    )
    request = validate_responses_payload(payload)
    params = build_anthropic_params(request)

    assert params.model == "claude-sonnet-4-5-20250929"
    assert params.tools and params.tools[0]["name"] == "get_weather"
    assert params.tool_choice == {
        "type": "tool",
        "name": "get_weather",
        "disable_parallel_tool_use": True,
    }
    assert params.thinking == {"type": "enabled", "budget_tokens": 3072}
    assert params.max_tokens == 128


def test_stream_events_from_anthropic_emits_required_action_and_tool_calls() -> None:
    events = [
        SimpleNamespace(
            type="content_block_start",
            content_block={"type": "tool_use", "name": "get_weather", "id": "toolu_123"},
            index=0,
        ),
        SimpleNamespace(
            type="content_block_delta",
            content_block={"type": "tool_use", "id": "toolu_123"},
            delta={"partial_json": '{"city":"San'},
            index=0,
        ),
        SimpleNamespace(
            type="content_block_delta",
            content_block={"type": "tool_use", "id": "toolu_123"},
            delta={"partial_json": ' Francisco"}'},
            index=0,
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(type="message_stop", stop_reason="tool_use"),
    ]

    stream_output = list(stream_events_from_anthropic(stream=events, model="claude-4"))
    parsed_events = [json.loads(item) for item in stream_output]
    assert parsed_events[0]["type"] == "response.created"
    required_action_event = parsed_events[-1]
    assert required_action_event["type"] == "response.required_action"
    calls = required_action_event["required_action"]["submit_tool_outputs"]["tool_calls"]
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"city": "San Francisco"}


def test_stream_events_from_anthropic_splits_reasoning_summary_from_output_text() -> None:
    events = [
        SimpleNamespace(
            type="content_block_delta",
            content_block={"type": "thinking"},
            delta={"text": "Reasoning step."},
            index=0,
        ),
        SimpleNamespace(
            type="content_block_delta",
            content_block={"type": "text"},
            delta={"text": "Final answer."},
            index=0,
        ),
        SimpleNamespace(type="message_stop", stop_reason="end_turn"),
    ]

    stream_output = list(stream_events_from_anthropic(stream=events, model="claude-4-sonnet"))
    parsed_events = [json.loads(item) for item in stream_output]

    reasoning_deltas = [
        event["delta"]
        for event in parsed_events
        if event["type"] == "response.reasoning_summary_text.delta"
    ]
    output_deltas = [
        event["delta"]
        for event in parsed_events
        if event["type"] == "response.output_text.delta"
    ]

    assert reasoning_deltas == ["Reasoning step."]
    assert output_deltas == ["Final answer."]

    completed = next(event for event in parsed_events if event["type"] == "response.completed")
    response_payload = completed["response"]

    assert response_payload["output_text"] == "Final answer."
    assert response_payload["reasoning_summary"] == "Reasoning step."

    content_items = response_payload["output"][0]["content"]
    assert {"type": "reasoning_summary", "text": "Reasoning step."} in content_items
