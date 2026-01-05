from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "responses" / "anthropic"


@dataclass(frozen=True)
class ResponsesFixture:
    name: str
    request: Mapping[str, object]
    anthropic_stream: Sequence[Mapping[str, object]]
    expected_events: Sequence[Mapping[str, object]]
    submit_tool_outputs: Mapping[str, object] | None = None


def load_fixture(name: str) -> ResponsesFixture:
    path = _FIXTURE_ROOT / f"{name}.json"
    data = json.loads(path.read_text())
    return ResponsesFixture(
        name=data["name"],
        request=data["request"],
        anthropic_stream=tuple(data["anthropic_stream"]),
        expected_events=tuple(data["expected_events"]),
        submit_tool_outputs=data.get("submit_tool_outputs"),
    )


def iter_fixtures() -> Iterator[ResponsesFixture]:
    for fixture_path in sorted(_FIXTURE_ROOT.glob("*.json")):
        data = json.loads(fixture_path.read_text())
        yield ResponsesFixture(
            name=data["name"],
            request=data["request"],
            anthropic_stream=tuple(data["anthropic_stream"]),
            expected_events=tuple(data["expected_events"]),
            submit_tool_outputs=data.get("submit_tool_outputs"),
        )
