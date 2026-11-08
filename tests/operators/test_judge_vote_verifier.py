"""Judge sees every candidate; MajorityVote normalizes and breaks ties as told;
Verifier reads verdicts strictly."""

from collections.abc import Callable
from typing import Any

import pytest

from ember.errors import ErrorKind, ModelError, TieError
from ember.operators import (
    Candidates,
    Check,
    Failed,
    Judge,
    MajorityVote,
    ModelCall,
    Succeeded,
    TiePolicy,
    Verdict,
    Verifier,
    fold_answer,
)
from ember.operators.verifier import read_verdict


def test_candidates_need_a_tuple_of_at_least_one_answer() -> None:
    with pytest.raises(ValueError, match="at least one answer"):
        Candidates("q", ())
    with pytest.raises(TypeError, match="tuple"):
        Candidates("q", ["a"])  # type: ignore[arg-type]


def test_the_judge_prompt_holds_the_question_and_every_candidate(
    fake_models: Callable[..., Any],
) -> None:
    fake = fake_models(lambda model, prompt: "Paris")
    judge = Judge(ModelCall("anthropic/claude-sonnet-4-5"), instructions="Pick the best.")
    result = judge(Candidates("Capital of France?", ("Paris", "Lyon", "Marseille")))
    assert result.text == "Paris"
    [sent] = fake.calls
    assert sent.prompt == (
        "Pick the best.\n\nQuestion:\nCapital of France?\n\n"
        "Answer 1:\nParis\n\nAnswer 2:\nLyon\n\nAnswer 3:\nMarseille"
    )


def test_vote_compares_normalized_answers_and_returns_the_first_spelling() -> None:
    vote = MajorityVote(normalize=fold_answer, tie=TiePolicy.RAISE)
    outcome = vote(Candidates("q", ("Lyon", " PARIS", "paris\n", "Paris")))
    assert outcome.answer == " PARIS"
    assert outcome.votes == 3
    assert outcome.tally == (("lyon", 1), ("paris", 3))


def test_exact_normalization_keeps_variants_apart() -> None:
    vote = MajorityVote(normalize=str, tie=TiePolicy.FIRST_SEEN)
    outcome = vote(Candidates("q", ("Paris", "paris", "Lyon", "Lyon")))
    assert (outcome.answer, outcome.votes) == ("Lyon", 2)


def test_ties_follow_the_declared_policy() -> None:
    tied = Candidates("q", ("b", "a", "a", "b", "c"))
    first = MajorityVote(normalize=str, tie=TiePolicy.FIRST_SEEN)(tied)
    assert (first.answer, first.votes) == ("b", 2)
    with pytest.raises(TieError) as raised:
        MajorityVote(normalize=str, tie=TiePolicy.RAISE)(tied)
    assert (raised.value.tied, raised.value.votes) == (("b", "a"), 2)
    assert str(raised.value) == "majority vote tied: 2 answers have 2 votes each"


def test_vote_has_no_default_policy() -> None:
    with pytest.raises(TypeError):
        MajorityVote()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("Looks right.\nVERDICT: PASS", Verdict.PASS),
        ("wrong year\n\nverdict: fail\n\n", Verdict.FAIL),
        ("VERDICT:PASS", Verdict.PASS),
        ("VERDICT: PASS, mostly", Verdict.UNREADABLE),
        ("VERDICT: PASS\nbut actually not", Verdict.UNREADABLE),
        ("", Verdict.UNREADABLE),
    ],
)
def test_verdicts_are_read_from_the_last_line_only(text: str, verdict: Verdict) -> None:
    assert read_verdict(text) is verdict


def test_verifier_checks_each_answer_and_records_operational_failures(
    fake_models: Callable[..., Any],
) -> None:
    overloaded = ModelError(ErrorKind.OVERLOADED, provider="openai", model="gpt-5", reason="x")

    def reply(model: str, prompt: str) -> str | ModelError:
        if prompt.endswith("Answer:\nLyon"):
            return "VERDICT: FAIL"
        if prompt.endswith("Answer:\nMarseille"):
            return overloaded
        return "VERDICT: PASS"

    fake = fake_models(reply)
    checks = Verifier(ModelCall("openai/gpt-5"))(Candidates("q", ("Paris", "Lyon", "Marseille")))
    assert [type(c) for c in checks] == [Succeeded, Succeeded, Failed]
    first, second, third = checks
    assert isinstance(first, Succeeded) and isinstance(first.value, Check)
    assert (first.value.answer, first.value.verdict) == ("Paris", Verdict.PASS)
    assert isinstance(second, Succeeded) and second.value.verdict is Verdict.FAIL
    assert third == Failed(2, overloaded)
    assert len(fake.calls) == 3


def test_verifier_propagates_fatal_errors(fake_models: Callable[..., Any]) -> None:
    denied = ModelError(ErrorKind.AUTH, provider="openai", model="gpt-5", reason="bad key")
    fake_models(lambda model, prompt: denied)
    with pytest.raises(ModelError) as raised:
        Verifier(ModelCall("openai/gpt-5"))(Candidates("q", ("a", "b")))
    assert raised.value is denied
