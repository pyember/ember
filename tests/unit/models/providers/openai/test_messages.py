from __future__ import annotations

import json

from ember.models.providers.openai._messages import (
    as_responses_turns,
    build_messages,
    normalize_context,
)


def test_normalize_context_preserves_structured_content() -> None:
    context = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }
    ]

    normalized = normalize_context(context)
    assert normalized is not None
    assert normalized[0]["role"] == "user"
    content = normalized[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"


def test_as_responses_turns_converts_text_to_input_text() -> None:
    messages = build_messages(
        "final prompt",
        system_prompt="system guidance",
        context=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "prior"},
                    {
                        "type": "tool_call",
                        "name": "weather",
                        "arguments": json.dumps({"loc": "sf"}),
                    },
                ],
            }
        ],
    )

    turns = as_responses_turns(messages)

    # System prompt becomes input_text content
    system_turn = turns[0]
    assert system_turn["role"] == "system"
    assert system_turn["content"] == [{"type": "input_text", "text": "system guidance"}]

    assistant_turn = turns[1]
    assistant_parts = assistant_turn["content"]
    assert any(part["type"] == "input_text" and part["text"] == "prior" for part in assistant_parts)
    assert any(part["type"] == "tool_call" for part in assistant_parts)

    user_turn = turns[2]
    assert user_turn["content"] == [{"type": "input_text", "text": "final prompt"}]
