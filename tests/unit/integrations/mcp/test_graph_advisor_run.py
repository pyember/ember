from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from ember.api.models import Response
from ember.models.schemas import ChatResponse, UsageStats
from integrations.mcp.graph_advisor.execution import run_session
from integrations.mcp.graph_advisor.schemas import Event, RunRequest
from integrations.mcp.graph_advisor.session import plan_session
from integrations.mcp.tools.model_tool import InvokeResult


@dataclass
class _FakeTool:
    results: List[InvokeResult]

    async def invoke(self, *, prompt: str, system: str | None, params: Dict[str, Any] | None):
        return self.results.pop(0)


class _FakeRegistry:
    def __init__(self, tools: Dict[str, _FakeTool]):
        self._tools = tools

    def has(self, model_id: str) -> bool:
        return model_id in self._tools

    def ensure_tool(self, model_id: str) -> _FakeTool:
        return self._tools[model_id]


def _make_result(
    text: str,
    *,
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
) -> InvokeResult:
    chat = ChatResponse(
        data=text,
        usage=UsageStats(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    response = Response(chat)
    return InvokeResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=0.0,
        model_id="fake",
        response=response,
    )


@pytest.mark.asyncio
async def test_run_session_emits_events():
    request = RunRequest(
        graph_spec=[
            "1E@fake-candidate",
            "1J@fake-judge",
        ],
        prompt="What is 2 + 2?",
    )

    session, plan_preview = plan_session(request)
    assert plan_preview.plan_summary.candidates == 1

    fake_registry = _FakeRegistry(
        {
            "fake-candidate": _FakeTool([_make_result("Candidate says 4.")]),
            "fake-judge": _FakeTool(
                [
                    _make_result(
                        json.dumps(
                            {
                                "synthesis": "Answer is 4",
                                "primary_source": 1,
                                "rationale": "basic math",
                            }
                        )
                    )
                ]
            ),
        }
    )

    events: List[Dict[str, Any]] = []

    async def emit(event: Event) -> None:
        events.append(event.model_dump())

    response = await run_session(session, request, fake_registry, emit)

    assert response.final_answer == "Answer is 4"
    event_types = [item["event"] for item in events]
    assert event_types.count("candidate") == 1
    assert "judge" in event_types
    assert event_types[-1] == "complete"
    assert response.usage.tokens_prompt >= 50
    assert response.telemetry.latency_ms >= 0
