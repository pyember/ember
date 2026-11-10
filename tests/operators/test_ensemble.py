"""Ensemble: per-member outcomes, a quorum, and an explicit failure partition."""

import dataclasses
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest

from ember import op
from ember.errors import ErrorKind, ModelError, QuorumError
from ember.llm.completion import Completion
from ember.llm.ref import ModelRef
from ember.llm.result import Result
from ember.operators import Ensemble, Failed, ModelCall, Operator, Succeeded
from ember.operators.ensemble import MEMBER_FAILURE_KINDS


def _error(kind: ErrorKind) -> ModelError:
    return ModelError(kind, provider="fake", model="m", reason="injected")


@dataclasses.dataclass(frozen=True, slots=True)
class Fails(Operator[str, str]):
    error: Exception

    def forward(self, input: str) -> str:
        raise self.error


@op
def echo(text: str) -> str:
    return text


def test_outcomes_are_in_member_order_with_values() -> None:
    @op
    def upper(text: str) -> str:
        return text.upper()

    outcomes = Ensemble((echo, upper), quorum=2)("a")
    assert outcomes == (Succeeded(0, "a"), Succeeded(1, "A"))


@pytest.mark.parametrize("kind", sorted(MEMBER_FAILURE_KINDS, key=lambda k: k.value))
def test_operational_failures_become_outcomes(kind: ErrorKind) -> None:
    error = _error(kind)
    outcomes = Ensemble((echo, Fails(error), echo), quorum=2)("a")
    assert outcomes == (Succeeded(0, "a"), Failed(1, error), Succeeded(2, "a"))


@pytest.mark.parametrize(
    "kind",
    [
        ErrorKind.AUTH,
        ErrorKind.PERMISSION,
        ErrorKind.QUOTA,
        ErrorKind.NOT_FOUND,
        ErrorKind.BAD_REQUEST,
        ErrorKind.MALFORMED,
        ErrorKind.UNPRICED,
    ],
)
def test_fatal_model_errors_propagate_even_when_quorum_would_hold(kind: ErrorKind) -> None:
    error = _error(kind)
    with pytest.raises(ModelError) as raised:
        Ensemble((echo, Fails(error), echo), quorum=1)("a")
    assert raised.value is error


def test_the_failure_partition_covers_every_error_kind() -> None:
    fatal = {
        ErrorKind.AUTH,
        ErrorKind.PERMISSION,
        ErrorKind.QUOTA,
        ErrorKind.NOT_FOUND,
        ErrorKind.BAD_REQUEST,
        ErrorKind.MALFORMED,
        ErrorKind.UNPRICED,
    }
    assert MEMBER_FAILURE_KINDS | fatal == set(ErrorKind)
    assert not MEMBER_FAILURE_KINDS & fatal


def test_bugs_in_user_code_propagate_unchanged() -> None:
    bug = TypeError("user bug")
    with pytest.raises(TypeError) as raised:
        Ensemble((echo, Fails(bug)), quorum=1)("a")
    assert raised.value is bug


def test_quorum_not_met_raises_with_every_failure() -> None:
    first, second = _error(ErrorKind.RATE_LIMITED), _error(ErrorKind.SERVER)
    with pytest.raises(QuorumError) as raised:
        Ensemble((Fails(first), echo, Fails(second)), quorum=2)("a")
    assert (raised.value.required, raised.value.succeeded) == (2, 1)
    assert raised.value.failures == ((0, first), (2, second))


def test_an_inner_quorum_failure_is_one_failed_member_of_the_outer_ensemble() -> None:
    inner = Ensemble((Fails(_error(ErrorKind.OVERLOADED)),), quorum=1)
    outer = Ensemble((inner, Ensemble((echo,), quorum=1)), quorum=1)
    first, second = outer("a")
    assert isinstance(first, Failed) and isinstance(first.error, QuorumError)
    assert second == Succeeded(1, (Succeeded(0, "a"),))


def test_construction_rejects_illegal_ensembles() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        Ensemble((), quorum=1)
    with pytest.raises(TypeError, match="tuple"):
        Ensemble([echo], quorum=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="@op"):
        Ensemble((echo, str.upper), quorum=1)  # type: ignore[arg-type]
    for quorum in (0, 3):
        with pytest.raises(ValueError, match="between 1 and 2"):
            Ensemble((echo, echo), quorum=quorum)
    with pytest.raises(TypeError):
        Ensemble((echo,))  # type: ignore[call-arg]  # quorum is required


@pytest.mark.parametrize("quorum", [True, 1.0, "1"])
def test_quorum_must_be_an_int(quorum: object) -> None:
    with pytest.raises(TypeError, match="quorum must be an int"):
        Ensemble((echo,), quorum=quorum)  # type: ignore[arg-type]


def test_quorum_shortfall_message_counts_and_names_failures() -> None:
    first = _error(ErrorKind.RATE_LIMITED)
    with pytest.raises(QuorumError) as raised:
        Ensemble((Fails(first), echo, Fails(first)), quorum=2)("a")
    message = str(raised.value)
    assert message.startswith("quorum not met: 1 succeeded, 2 errored; 2 required; member 0: ")
    assert "; member 2: " in message


def test_replace_cannot_produce_an_invalid_ensemble() -> None:
    ensemble = Ensemble((echo, echo), quorum=2)
    with pytest.raises(ValueError):
        dataclasses.replace(ensemble, members=(echo,))


def test_model_call_members_send_one_request_each(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: f"{model}:{prompt}")
    members = (ModelCall("openai/gpt-5"), ModelCall("anthropic/claude-sonnet-4-5"))
    outcomes = Ensemble(members, quorum=2)("q")
    texts = [o.value.text for o in outcomes if isinstance(o, Succeeded)]
    assert texts == ["openai/gpt-5:q", "anthropic/claude-sonnet-4-5:q"]
    assert len(fake.calls) == 2


@dataclasses.dataclass(frozen=True, slots=True)
class FailsWhenRead(Operator[str, Completion]):
    """Returns a pending Completion whose call failed, as members do inside jit."""

    error: ModelError

    def forward(self, input: str) -> Completion:
        future: Future[Result] = Future()
        future.set_exception(self.error)
        return Completion(future, ModelRef("fake", "m"))


def test_a_pending_failure_is_the_members_outcome_not_a_later_surprise() -> None:
    error = _error(ErrorKind.OVERLOADED)
    outcomes = Ensemble((echo, FailsWhenRead(error)), quorum=1)("a")
    assert outcomes == (Succeeded(0, "a"), Failed(1, error))


def test_a_pending_fatal_failure_propagates_from_the_ensemble() -> None:
    error = _error(ErrorKind.AUTH)
    with pytest.raises(ModelError) as raised:
        Ensemble((echo, FailsWhenRead(error)), quorum=1)("a")
    assert raised.value is error


def test_succeeded_model_calls_hold_finished_completions(fake_models: Callable[..., Any]) -> None:
    fake_models(lambda model, prompt: "done")
    [outcome] = Ensemble((ModelCall("openai/gpt-5"),), quorum=1)("q")
    assert isinstance(outcome, Succeeded) and isinstance(outcome.value, Completion)
    assert outcome.value.done() and outcome.value.text == "done"
