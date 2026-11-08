"""Running a NON graph.

`build_graph` returns a `NonGraph`: an operator from a question to an `Answer`.
A run first checks every model stage with the active client, as a call would
before sending, so a bad model or setting in any stage fails before the first
request. Then it compiles the parsed stages to the operators in
`ember.operators`:

- The E stages run concurrently. Each makes ``count`` calls at once, as an
  `Ensemble` of `ModelCall`s does. Their usable answers, in stage order and
  then member order, are the candidates.
- A J stage makes ``count`` `Judge` calls at once, each reading the question
  and every candidate. Their usable answers become the candidates.
- A V stage is a `Verifier`. The answers that pass become the candidates.
- An M stage is a `MajorityVote` comparing answers by `fold_answer` and
  settling ties by `TiePolicy.FIRST_SEEN`. The winner is the only candidate.

An answer is usable only when its call finished with `FinishReason.STOP`; a
refusal, a filtered, truncated or tool-call reply is kept in the stage's
record but is never a candidate. An E or J stage with a ``seed`` gives member
``i`` the seed ``seed + i``, so no two members send the same request. A stage
with fewer usable answers (or, for V, passing answers) than its quorum raises
`QuorumError`. The answer is the one candidate the last stage leaves, and
`Answer.stages` records every stage's outcomes beside it.
"""

import dataclasses
import functools
from collections.abc import Callable

from ember import xcs
from ember.errors import EmberError, QuorumError
from ember.llm.catalog import Catalog
from ember.llm.client import active
from ember.llm.completion import Completion
from ember.llm.result import FinishReason
from ember.non.grammar import ModelStage, Role, Stage, VoteStage, parse
from ember.non.plan import Plan, plan
from ember.operators.base import Operator
from ember.operators.candidates import Candidates
from ember.operators.ensemble import Failed, Outcome, Succeeded, run_members
from ember.operators.judge import Judge
from ember.operators.model_call import ModelCall
from ember.operators.verifier import Check, Verdict, Verifier
from ember.operators.vote import MajorityVote, TiePolicy, Vote, fold_answer

__all__ = ["Answer", "NonGraph", "Sampled", "StageRecord", "Verified", "Voted", "build_graph"]


@dataclasses.dataclass(frozen=True, slots=True)
class Sampled:
    """What an E or J stage produced: one outcome per call, in member order."""

    stage: ModelStage
    outcomes: tuple[Outcome[Completion], ...]

    @property
    def answers(self) -> tuple[str, ...]:
        """The texts of the calls that finished with `FinishReason.STOP`."""
        return tuple(
            o.value.text
            for o in self.outcomes
            if isinstance(o, Succeeded) and o.value.finish_reason is FinishReason.STOP
        )

    @property
    def rejected(self) -> tuple[tuple[int, str], ...]:
        """``(member, reason)`` for each call that answered but is not usable."""
        return tuple(
            (o.member, f"unusable ({o.value.finish_reason.value})")
            for o in self.outcomes
            if isinstance(o, Succeeded) and o.value.finish_reason is not FinishReason.STOP
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Verified:
    """What a V stage produced: one check per candidate, in candidate order."""

    stage: ModelStage
    checks: tuple[Outcome[Check], ...]

    @property
    def answers(self) -> tuple[str, ...]:
        """The answers whose check passed."""
        return tuple(
            o.value.answer
            for o in self.checks
            if isinstance(o, Succeeded) and o.value.verdict is Verdict.PASS
        )

    @property
    def rejected(self) -> tuple[tuple[int, str], ...]:
        """``(answer, reason)`` for each answer the verifier did not pass."""
        reasons = {Verdict.FAIL: "failed", Verdict.UNREADABLE: "unreadable"}
        return tuple(
            (o.member, reasons[o.value.verdict])
            for o in self.checks
            if isinstance(o, Succeeded) and o.value.verdict is not Verdict.PASS
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Voted:
    """What an M stage produced."""

    stage: VoteStage
    vote: Vote

    @property
    def answers(self) -> tuple[str, ...]:
        """The winning answer."""
        return (self.vote.answer,)


StageRecord = Sampled | Verified | Voted


@dataclasses.dataclass(frozen=True, slots=True)
class Answer:
    """A NON graph's answer and the record of how it was reached.

    Attributes:
        text: The answer.
        stages: One record per stage, in specification order.
    """

    text: str
    stages: tuple[StageRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class NonGraph(Operator[str, Answer]):
    """A runnable NON graph. Build one with `build_graph`.

    Attributes:
        spec: The specification. It is parsed when the graph is constructed,
            so an invalid one raises `NonSyntaxError` there, and again on each
            run.
    """

    spec: str

    def __post_init__(self) -> None:
        parse(self.spec)

    def stages(self) -> tuple[Stage, ...]:
        """Returns the parsed stages."""
        return parse(self.spec)

    def plan(self, catalog: Catalog) -> Plan:
        """Returns upper bounds for one run; see `ember.non.plan`.

        Args:
            catalog: Where model limits and rates come from, usually
                ``ember.llm.catalog.default()``.
        """
        return plan(self.stages(), catalog)

    def forward(self, input: str) -> Answer:
        if not isinstance(input, str):
            raise TypeError(f"a NON graph takes a question string, got {type(input)}")
        stages = self.stages()
        _check_before_sending(stages)
        generation = [s for s in stages if isinstance(s, ModelStage) and s.role is Role.ENSEMBLE]
        records: list[StageRecord] = list(xcs.map(functools.partial(_generate, input), generation))
        answers = tuple(answer for record in records for answer in record.answers)
        for stage in stages[len(generation) :]:
            record = _refine(stage, Candidates(input, answers))
            records.append(record)
            answers = record.answers
        return Answer(text=answers[0], stages=tuple(records))


def build_graph(spec: str) -> NonGraph:
    """Parses ``spec`` and returns the graph as a runnable operator.

    Example:
        >>> from ember import build_graph
        >>> graph = build_graph("3E@openai/gpt-5.5, 1J@anthropic/claude-opus-5-5")
        >>> [stage.column for stage in graph.stages()]
        [0, 19]

    Raises:
        NonSyntaxError: ``spec`` is not a valid graph.
    """
    return NonGraph(spec)


def _check_before_sending(stages: tuple[Stage, ...]) -> None:
    """Raises, for the first stage that would fail, what its calls would raise before sending."""
    client = active()
    for stage in stages:
        if isinstance(stage, ModelStage):
            client.check(stage.model, sampling=stage.params.sampling)


def _generate(question: str, stage: ModelStage) -> Sampled:
    calls: list[Callable[[], Completion]] = [
        functools.partial(call, question) for call in _member_calls(stage)
    ]
    return _require_usable(Sampled(stage, run_members(calls)))


def _refine(stage: Stage, candidates: Candidates) -> StageRecord:
    if isinstance(stage, VoteStage):
        return Voted(
            stage, MajorityVote(normalize=fold_answer, tie=TiePolicy.FIRST_SEEN)(candidates)
        )
    if stage.role is Role.JUDGE:
        calls: list[Callable[[], Completion]] = [
            functools.partial(Judge(call), candidates) for call in _member_calls(stage)
        ]
        return _require_usable(Sampled(stage, run_members(calls)))
    if stage.role is Role.VERIFIER:
        return _require_passed(Verified(stage, Verifier(_member_calls(stage)[0])(candidates)))
    raise AssertionError(f"parse keeps E stages before every other stage, found {stage}")


def _member_calls(stage: ModelStage) -> list[ModelCall]:
    """One call per member; with a seed, member ``i`` uses ``seed + i``."""
    params = stage.params
    sampling = params.sampling
    base = ModelCall(stage.model, system=params.system, sampling=sampling)
    if sampling.seed is None:
        return [base] * stage.count
    return [
        dataclasses.replace(base, sampling=dataclasses.replace(sampling, seed=sampling.seed + i))
        for i in range(stage.count)
    ]


def _require_usable(record: Sampled) -> Sampled:
    required = _quorum(record.stage, default=record.stage.count)
    if len(record.answers) < required:
        raise QuorumError(
            required=required,
            succeeded=len(record.answers),
            counted_as="usable",
            rejected=record.rejected,
            failures=_failures(record.outcomes),
        )
    return record


def _require_passed(record: Verified) -> Verified:
    required = _quorum(record.stage, default=1)
    if len(record.answers) < required:
        raise QuorumError(
            required=required,
            succeeded=len(record.answers),
            counted_as="passed",
            rejected=record.rejected,
            failures=_failures(record.checks),
        )
    return record


def _failures(outcomes: tuple[Outcome[object], ...]) -> tuple[tuple[int, EmberError], ...]:
    return tuple((o.member, o.error) for o in outcomes if isinstance(o, Failed))


def _quorum(stage: ModelStage, *, default: int) -> int:
    return default if stage.params.quorum is None else stage.params.quorum
