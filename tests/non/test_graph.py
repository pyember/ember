"""NON graphs run: build_graph returns an operator, and every stage executes."""

import collections
import dataclasses
import itertools
from collections.abc import Callable
from typing import Any

import pytest

from ember import build_graph
from ember.errors import ErrorKind, ModelError, NonSyntaxError, QuorumError
from ember.non import NonGraph, Sampled, Verified, Voted
from ember.operators import Ensemble, Failed, Operator, Succeeded


def _error(kind: ErrorKind, model: str) -> ModelError:
    provider, _, name = model.partition("/")
    return ModelError(kind, provider=provider, model=name, reason="injected")


def test_ensemble_then_judge_answers_and_the_judge_sees_every_candidate(
    fake_models: Callable[..., Any],
) -> None:
    numbers = itertools.count(1)

    def reply(model: str, prompt: str) -> str:
        return "final" if model == "anthropic/claude-sonnet-4-5" else f"draft {next(numbers)}"

    fake = fake_models(reply)
    graph = build_graph("5E@openai/gpt-5(temperature=0.7), 1J@anthropic/claude-sonnet-4-5")
    answer = graph("What is 2 + 2?")

    assert answer.text == "final"
    generated, judged = answer.stages
    assert isinstance(generated, Sampled) and isinstance(judged, Sampled)
    assert generated.answers == tuple(f"draft {n}" for n in range(1, 6))
    assert judged.answers == ("final",)
    sent = fake.calls
    assert [call.model for call in sent] == ["openai/gpt-5"] * 5 + ["anthropic/claude-sonnet-4-5"]
    assert {call.temperature for call in sent[:5]} == {0.7}
    assert all(f"Answer {n}:\ndraft {n}" in sent[5].prompt for n in range(1, 6))
    assert "Question:\nWhat is 2 + 2?" in sent[5].prompt


def test_majority_vote_picks_the_most_common_answer(fake_models: Callable[..., Any]) -> None:
    replies = iter(["Paris", "Lyon", "paris", " PARIS ", "Lyon"])
    fake_models(lambda model, prompt: next(replies))
    answer = build_graph("5E@openai/gpt-5, 1M")("Capital of France?")
    assert answer.text == "Paris"
    voted = answer.stages[-1]
    assert isinstance(voted, Voted)
    assert voted.vote.tally == (("paris", 3), ("lyon", 2))


def test_verifier_filters_what_the_judge_sees(fake_models: Callable[..., Any]) -> None:
    drafts = iter(["good one", "bad one", "good two"])

    def reply(model: str, prompt: str) -> str:
        if model == "x/verifier":
            return "VERDICT: FAIL" if prompt.endswith("bad one") else "VERDICT: PASS"
        if model == "x/judge":
            return "judged"
        return next(drafts)

    fake = fake_models(reply)
    answer = build_graph("3E@x/gen, 1V@x/verifier, 1J@x/judge")("q")
    assert answer.text == "judged"
    verified = answer.stages[1]
    assert isinstance(verified, Verified) and verified.answers == ("good one", "good two")
    judge_prompt = fake.calls[-1].prompt
    assert "good one" in judge_prompt and "good two" in judge_prompt
    assert "bad one" not in judge_prompt


def test_e_stages_pool_their_answers_in_stage_then_member_order(
    fake_models: Callable[..., Any],
) -> None:
    counters: collections.defaultdict[str, itertools.count[int]] = collections.defaultdict(
        itertools.count
    )
    fake_models(lambda model, prompt: f"{model}#{next(counters[model])}")
    answer = build_graph("2E@a/x, 1E@b/y, 3J@c/z, 1M")("q")
    first, second, judged, _ = answer.stages
    assert isinstance(first, Sampled) and first.answers == ("a/x#0", "a/x#1")
    assert isinstance(second, Sampled) and second.answers == ("b/y#0",)
    assert isinstance(judged, Sampled) and len(judged.outcomes) == 3
    assert answer.text == "c/z#0"


def test_quorum_absorbs_operational_failures_and_records_them(
    fake_models: Callable[..., Any],
) -> None:
    calls = itertools.count()
    limited = _error(ErrorKind.RATE_LIMITED, "a/b")

    def reply(model: str, prompt: str) -> str | ModelError:
        return limited if next(calls) == 1 else "ok"

    fake_models(reply)
    answer = build_graph("3E@a/b(quorum=2), 1M")("q")
    generated = answer.stages[0]
    assert isinstance(generated, Sampled)
    assert generated.outcomes[1] == Failed(1, limited)
    assert answer.text == "ok"


def test_quorum_shortfall_raises(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: _error(ErrorKind.SERVER, model))
    with pytest.raises(QuorumError) as raised:
        build_graph("3E@a/b(quorum=1), 1M")("q")
    assert (raised.value.required, raised.value.succeeded) == (1, 0)
    assert [index for index, _ in raised.value.failures] == [0, 1, 2]


def test_verifier_quorum_shortfall_raises(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: "VERDICT: FAIL" if model == "v/v" else "draft")
    with pytest.raises(QuorumError) as raised:
        build_graph("2E@a/b, 1V@v/v, 1J@j/j")("q")
    assert (raised.value.required, raised.value.succeeded) == (1, 0)


def test_fatal_errors_propagate_unchanged(fake_models: Callable[..., Any]) -> None:
    denied = _error(ErrorKind.AUTH, "a/b")
    fake_models(lambda model, prompt: denied)
    with pytest.raises(ModelError) as raised:
        build_graph("3E@a/b(quorum=1), 1M")("q")
    assert raised.value is denied


def test_a_graph_is_an_operator_and_nests_in_an_ensemble(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: model)
    graphs = (build_graph("1E@a/b, 1M"), build_graph("1E@c/d, 1M"))
    assert all(isinstance(graph, Operator) for graph in graphs)
    outcomes = Ensemble(graphs, quorum=2)("q")
    assert [o.value.text for o in outcomes if isinstance(o, Succeeded)] == ["a/b", "c/d"]


def test_invalid_specs_fail_at_build_time_including_through_replace() -> None:
    with pytest.raises(NonSyntaxError):
        build_graph("3E@a/b")
    graph = build_graph("3E@a/b, 1M")
    with pytest.raises(NonSyntaxError):
        dataclasses.replace(graph, spec="0E@a/b, 1M")


def test_a_graph_takes_only_a_question_string(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: "x")
    with pytest.raises(TypeError, match="question string"):
        NonGraph("1E@a/b, 1M")({"query": "q"})  # type: ignore[arg-type]
    assert fake.calls == []
