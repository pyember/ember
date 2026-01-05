from __future__ import annotations

import json
from typing import Iterable, Mapping

import pytest

from ember.models.providers.anthropic_responses import stream_events_from_anthropic
from tests.helpers.responses_fixtures import iter_fixtures, load_fixture


def _decode_events(events: Iterable[str]) -> list[dict[str, object]]:
    return [json.loads(event) for event in events]


def _assert_superset(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for key, value in expected.items():
        assert key in actual, f"Missing key '{key}' in event {actual}"
        if isinstance(value, Mapping):
            assert isinstance(actual[key], Mapping), f"{key} should be mapping"
            _assert_superset(actual[key], value)  # type: ignore[arg-type]
        else:
            assert actual[key] == value, f"{key} mismatch: {actual[key]} != {value}"


@pytest.mark.parametrize("fixture_name", [fixture.name for fixture in iter_fixtures()])
def test_anthropic_stream_matches_fixture(fixture_name: str) -> None:
    fixture = load_fixture(fixture_name)
    actual_events = _decode_events(
        stream_events_from_anthropic(
            stream=fixture.anthropic_stream,
            model=str(fixture.request["model"]),
            response_id="resp_fixture",
        )
    )
    assert len(actual_events) == len(fixture.expected_events), (
        fixture.name,
        actual_events,
    )
    for actual, expected in zip(actual_events, fixture.expected_events, strict=True):
        _assert_superset(actual, expected)


def test_submit_tool_outputs_fixture_round_trip() -> None:
    fixture = load_fixture("text_with_tool_call")
    assert fixture.submit_tool_outputs is not None
    events = _decode_events(
        stream_events_from_anthropic(
            stream=fixture.submit_tool_outputs["anthropic_stream"],
            model=str(fixture.request["model"]),
            response_id="resp_fixture",
        )
    )
    expected_events = fixture.submit_tool_outputs["expected_events"]
    assert len(events) == len(expected_events)
    for actual, expected in zip(events, expected_events, strict=True):
        _assert_superset(actual, expected)
