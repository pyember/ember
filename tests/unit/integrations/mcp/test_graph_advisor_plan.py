from __future__ import annotations

from integrations.mcp.graph_advisor.schemas import RunRequest
from integrations.mcp.graph_advisor.session import plan_session


def test_plan_session_normalizes_spec():
    request = RunRequest(
        graph_spec=[
            "1E@fake-candidate",
            "1J@fake-judge",
        ],
        prompt="ping",
    )

    session, response = plan_session(request)

    assert response.plan_summary.candidates == 1
    assert response.plan_summary.judge == 1
    assert session.plan.candidates[0].model == "fake-candidate"
    assert session.plan.judge.model == "fake-judge"
    assert response.estimated_usage.tokens_prompt >= 400

