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
from ember.llm.fake import Reply
from ember.llm.result import FinishReason
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


def _first_n_calls(n: int, first: Reply | str, rest: Reply | str) -> Callable[..., Reply | str]:
    """A reply that gives ``first`` to the first ``n`` calls to arrive and ``rest`` after."""
    lock = threading.Lock()
    seen: list[None] = []

    def reply(model: str, prompt: str) -> Reply | str:
        with lock:
            seen.append(None)
            early = len(seen) <= n
        return first if early else rest

    return reply


@pytest.mark.parametrize(
    "finish",
    [
        FinishReason.CONTENT_FILTER,
        FinishReason.REFUSAL,
        FinishReason.LENGTH,
        FinishReason.TOOL_CALLS,
    ],
)
def test_only_answers_that_stopped_become_candidates(
    fake_models: Callable[..., Any], finish: FinishReason
) -> None:
    fake_models(_first_n_calls(2, Reply("", finish_reason=finish), "Paris"))
    answer = build_graph("3E@openai/m(quorum=1), 1M")("Capital of France?")
    assert answer.text == "Paris"
    generated = answer.stages[0]
    assert isinstance(generated, Sampled)
    assert generated.answers == ("Paris",)
    unusable = [o for o in generated.outcomes if isinstance(o, Succeeded)]
    assert sorted(o.value.finish_reason.value for o in unusable) == sorted(
        [finish.value, finish.value, "stop"]
    )


def test_unusable_answers_count_against_the_quorum(fake_models: Callable[..., Any]) -> None:
    fake_models(_first_n_calls(2, Reply("", finish_reason=FinishReason.CONTENT_FILTER), "Paris"))
    with pytest.raises(QuorumError) as raised:
        build_graph("3E@openai/m(quorum=2), 1M")("q")
    assert (raised.value.required, raised.value.succeeded) == (2, 1)
    assert sorted(reason for _, reason in raised.value.rejected) == [
        "unusable (content_filter)",
        "unusable (content_filter)",
    ]
    assert str(raised.value).startswith(
        "quorum not met: 1 usable, 2 unusable (content_filter); 2 required"
    )


def test_a_judge_that_refuses_is_not_an_answer(fake_models: Callable[..., Any]) -> None:
    refusal = Reply("I can't help with that.", finish_reason=FinishReason.REFUSAL)
    fake_models(lambda model, prompt: refusal if model == "anthropic/j" else "draft")
    with pytest.raises(QuorumError, match=r"0 usable, 1 unusable \(refusal\); 1 required"):
        build_graph("2E@openai/m, 1J@anthropic/j")("q")


def test_a_seed_is_derived_per_member(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: "same")
    build_graph("3E@openai/m(seed=7), 1E@openai/n, 2J@anthropic/j(seed=0), 1M")("q")
    seeds: dict[str, list[int | None]] = {}
    for call in fake.calls:
        seeds.setdefault(call.model, []).append(call.seed)
    assert sorted(seeds["openai/m"]) == [7, 8, 9]  # type: ignore[type-var]
    assert seeds["openai/n"] == [None]
    assert sorted(seeds["anthropic/j"]) == [0, 1]  # type: ignore[type-var]


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        ("2E@openai/m, 1J@openai/gpt-5.5(temperature=0.5)", ValueError),
        ("2E@openai/m, 1V@openai/gpt-5.5(reasoning_effort=enormous), 1M", ValueError),
        ("2E@openai/m, 1J@nowhere/judge", ModelError),
    ],
)
def test_every_stage_is_checked_before_anything_is_sent(
    fake_models: Callable[..., Any], spec: str, error: type[Exception]
) -> None:
    fake = fake_models(lambda model, prompt: "x")
    with pytest.raises(error):
        build_graph(spec)("q")
    assert fake.provider.requests == ()


def test_a_verifier_shortfall_counts_passed_failed_and_unreadable(
    fake_models: Callable[..., Any],
) -> None:
    def reply(model: str, prompt: str) -> str:
        if model != "anthropic/v":
            return model.removeprefix("openai/")
        if prompt.endswith("Answer:\nfail"):
            return "VERDICT: FAIL"
        if prompt.endswith("Answer:\nmumble"):
            return "verdict: pass"
        return "VERDICT: PASS"

    fake_models(reply)
    spec = "1E@openai/ok, 1E@openai/fail, 1E@openai/mumble, 1V@anthropic/v(quorum=2), 1M"
    with pytest.raises(QuorumError) as raised:
        build_graph(spec)("q")
    assert str(raised.value) == "quorum not met: 1 passed, 1 failed, 1 unreadable; 2 required"
