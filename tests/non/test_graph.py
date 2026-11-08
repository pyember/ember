"""NON graphs run: build_graph returns an operator, and every stage executes.

Calls within a stage run concurrently, so every reply here is a function of
the model and the prompt, never of arrival order.
"""

import dataclasses
import threading
from collections.abc import Callable
from typing import Any

import pytest

from ember import build_graph
from ember.errors import ErrorKind, ModelError, NonSyntaxError, QuorumError
from ember.non import NonGraph, Sampled, Verified, Voted
from ember.operators import Ensemble, Failed, Operator, Succeeded


def test_ensembles_then_judge_and_the_judge_sees_every_candidate_in_order(
    fake_models: Callable[..., Any],
) -> None:
    def reply(model: str, prompt: str) -> str:
        return "final" if model == "anthropic/judge" else f"draft from {model}"

    fake = fake_models(reply)
    graph = build_graph("3E@openai/writer(temperature=0.7), 2E@gemini/writer, 1J@anthropic/judge")
    answer = graph("What is 2 + 2?")

    assert answer.text == "final"
    first, second, judged = answer.stages
    assert isinstance(first, Sampled) and isinstance(second, Sampled)
    assert first.answers == ("draft from openai/writer",) * 3
    assert second.answers == ("draft from gemini/writer",) * 2
    assert isinstance(judged, Sampled) and judged.answers == ("final",)
    sent = fake.calls
    assert sorted(call.model for call in sent[:5]) == ["gemini/writer"] * 2 + ["openai/writer"] * 3
    assert sent[5].model == "anthropic/judge"
    assert {call.temperature for call in sent if call.model == "openai/writer"} == {0.7}
    judge_prompt = sent[5].prompt
    assert isinstance(judge_prompt, str)
    assert "Question:\nWhat is 2 + 2?" in judge_prompt
    for number, model in enumerate(["openai/writer"] * 3 + ["gemini/writer"] * 2, start=1):
        assert f"Answer {number}:\ndraft from {model}" in judge_prompt


def test_majority_vote_compares_normalized_answers(fake_models: Callable[..., Any]) -> None:
    texts = {"openai/a": "Paris", "openai/b": "Lyon", "openai/c": " PARIS "}
    fake_models(lambda model, prompt: texts[model])
    answer = build_graph("2E@openai/a, 2E@openai/b, 1E@openai/c, 1M")("Capital of France?")
    assert answer.text == "Paris"
    voted = answer.stages[-1]
    assert isinstance(voted, Voted)
    assert voted.vote.tally == (("paris", 3), ("lyon", 2))


def test_verifier_filters_what_the_judge_sees(fake_models: Callable[..., Any]) -> None:
    def reply(model: str, prompt: str) -> str:
        if model == "anthropic/verifier":
            return "VERDICT: FAIL" if prompt.endswith("bad") else "VERDICT: PASS"
        if model == "gemini/judge":
            return "judged"
        return model.removeprefix("openai/")

    fake = fake_models(reply)
    spec = "1E@openai/good, 1E@openai/bad, 1E@openai/fine, 1V@anthropic/verifier, 1J@gemini/judge"
    answer = build_graph(spec)("q")
    assert answer.text == "judged"
    verified = answer.stages[3]
    assert isinstance(verified, Verified) and verified.answers == ("good", "fine")
    [judge_call] = [call for call in fake.calls if call.model == "gemini/judge"]
    assert isinstance(judge_call.prompt, str)
    assert "Answer 1:\ngood\n\nAnswer 2:\nfine" in judge_call.prompt
    assert "bad" not in judge_call.prompt


def test_several_judges_then_a_vote(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: model)
    answer = build_graph("2E@openai/x, 1E@anthropic/y, 3J@gemini/z, 1M")("q")
    first, second, judged, voted = answer.stages
    assert isinstance(first, Sampled) and first.answers == ("openai/x", "openai/x")
    assert isinstance(second, Sampled) and second.answers == ("anthropic/y",)
    assert isinstance(judged, Sampled) and judged.answers == ("gemini/z",) * 3
    assert isinstance(voted, Voted) and answer.text == "gemini/z"
    assert len(fake.calls) == 6


def test_quorum_absorbs_operational_failures_and_records_them(
    fake_models: Callable[..., Any],
) -> None:
    lock = threading.Lock()
    failed: list[bool] = []

    def reply(model: str, prompt: str) -> str | ErrorKind:
        with lock:  # Exactly one call fails; which one depends on arrival.
            first = not failed
            failed.append(True)
        return ErrorKind.RATE_LIMITED if first else "ok"

    fake_models(reply)
    answer = build_graph("3E@openai/m(quorum=2), 1M")("q")
    generated = answer.stages[0]
    assert isinstance(generated, Sampled)
    failures = [o for o in generated.outcomes if isinstance(o, Failed)]
    assert [f.error.kind for f in failures] == [ErrorKind.RATE_LIMITED]
    assert answer.text == "ok"


def test_quorum_shortfall_raises(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: ErrorKind.SERVER)
    with pytest.raises(QuorumError) as raised:
        build_graph("3E@openai/m(quorum=1), 1M")("q")
    assert (raised.value.required, raised.value.succeeded) == (1, 0)
    assert [index for index, _ in raised.value.failures] == [0, 1, 2]


def test_verifier_quorum_shortfall_raises(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: "VERDICT: FAIL" if model == "anthropic/v" else "draft")
    with pytest.raises(QuorumError) as raised:
        build_graph("2E@openai/m, 1V@anthropic/v, 1J@gemini/j")("q")
    assert (raised.value.required, raised.value.succeeded) == (1, 0)


def test_fatal_errors_propagate(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: ErrorKind.AUTH)
    with pytest.raises(ModelError) as raised:
        build_graph("3E@openai/m(quorum=1), 1M")("q")
    assert raised.value.kind is ErrorKind.AUTH


def test_an_unknown_provider_is_fatal_before_anything_is_sent(
    fake_models: Callable[..., Any],
) -> None:
    fake = fake_models(lambda model, prompt: "x")
    with pytest.raises(ModelError) as raised:
        build_graph("2E@nowhere/m, 1M")("q")
    assert raised.value.kind is ErrorKind.NOT_FOUND
    assert fake.calls == []


def test_a_graph_is_an_operator_and_nests_in_an_ensemble(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: model)
    graphs = (build_graph("1E@openai/a, 1M"), build_graph("1E@anthropic/b, 1M"))
    assert all(isinstance(graph, Operator) for graph in graphs)
    outcomes = Ensemble(graphs, quorum=2)("q")
    assert [o.value.text for o in outcomes if isinstance(o, Succeeded)] == [
        "openai/a",
        "anthropic/b",
    ]


def test_invalid_specs_fail_at_build_time_including_through_replace() -> None:
    with pytest.raises(NonSyntaxError):
        build_graph("3E@openai/m")
    graph = build_graph("3E@openai/m, 1M")
    with pytest.raises(NonSyntaxError):
        dataclasses.replace(graph, spec="0E@openai/m, 1M")


def test_a_graph_takes_only_a_question_string(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: "x")
    with pytest.raises(TypeError, match="question string"):
        NonGraph("1E@openai/m, 1M")({"query": "q"})  # type: ignore[arg-type]
    assert fake.calls == []
