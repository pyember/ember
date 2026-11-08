"""Running a NON graph.

`build_graph` returns a `NonGraph`: an operator from a question to an `Answer`.
Each run compiles the parsed stages to the operators in `ember.operators`:

- The E stages run concurrently. Each is an `Ensemble` of ``count`` identical
  `ModelCall`s with quorum ``quorum`` (default: all of them). Their answers, in
  stage order and then member order, are the candidates.
- A J stage is an `Ensemble` of ``count`` `Judge`s, each reading the question
  and every candidate. Their answers become the candidates.
- A V stage is a `Verifier`. The answers that pass become the candidates; fewer
  than ``quorum`` (default 1) passing raises `QuorumError`.
- An M stage is a `MajorityVote` comparing answers by `fold_answer` and
  settling ties by `TiePolicy.FIRST_SEEN`. The winner is the only candidate.

The answer is the one candidate the last stage leaves. `Answer.stages` records
every stage's outcomes beside it.
"""

import dataclasses
import functools
from collections.abc import Callable

from ember.errors import QuorumError
from ember.llm.completion import Completion
from ember.llm.ref import ModelRef
from ember.llm.spec import ModelSpec
from ember.non.grammar import ModelStage, Role, Stage, VoteStage, parse
from ember.non.plan import Plan, plan
from ember.operators.base import Operator
from ember.operators.candidates import Candidates
from ember.operators.ensemble import Ensemble, Failed, Outcome, Succeeded
from ember.operators.judge import Judge
from ember.operators.model_call import ModelCall
from ember.operators.upstream import xcs_map
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
        """The texts of the calls that succeeded."""
        return tuple(o.value.text for o in self.outcomes if isinstance(o, Succeeded))


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
            run, which costs microseconds.
    """

    spec: str

    def __post_init__(self) -> None:
        parse(self.spec)

    def stages(self) -> tuple[Stage, ...]:
        """Returns the parsed stages."""
        return parse(self.spec)

    def plan(self, resolve: Callable[[ModelRef], ModelSpec]) -> Plan:
        """Returns upper bounds for one run; see `ember.non.plan`.

        Args:
            resolve: Returns the catalog spec for a model.
        """
        return plan(self.stages(), resolve)

    def forward(self, input: str) -> Answer:
        if not isinstance(input, str):
            raise TypeError(f"a NON graph takes a question string, got {type(input)}")
        stages = self.stages()
        generation = [s for s in stages if isinstance(s, ModelStage) and s.role is Role.ENSEMBLE]
        records: list[StageRecord] = list(xcs_map(functools.partial(_generate, input), generation))
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
        >>> graph = build_graph("3E@openai/gpt-5(temperature=0.7), 1J@anthropic/claude-sonnet-4-5")
        >>> [stage.column for stage in graph.stages()]
        [0, 34]

    Raises:
        NonSyntaxError: ``spec`` is not a valid graph.
    """
    return NonGraph(spec)


def _generate(question: str, stage: ModelStage) -> Sampled:
    call = _model_call(stage)
    ensemble = Ensemble((call,) * stage.count, quorum=_quorum(stage, default=stage.count))
    return Sampled(stage, ensemble(question))


def _refine(stage: Stage, candidates: Candidates) -> StageRecord:
    if isinstance(stage, VoteStage):
        return Voted(
            stage, MajorityVote(normalize=fold_answer, tie=TiePolicy.FIRST_SEEN)(candidates)
        )
    if stage.role is Role.JUDGE:
        judge = Judge(_model_call(stage))
        judges = Ensemble((judge,) * stage.count, quorum=_quorum(stage, default=stage.count))
        return Sampled(stage, judges(candidates))
    if stage.role is Role.VERIFIER:
        return _verify(stage, candidates)
    raise AssertionError(f"parse keeps E stages before every other stage, found {stage}")


def _verify(stage: ModelStage, candidates: Candidates) -> Verified:
    record = Verified(stage, Verifier(_model_call(stage))(candidates))
    required = _quorum(stage, default=1)
    if len(record.answers) < required:
        failures = tuple((o.member, o.error) for o in record.checks if isinstance(o, Failed))
        raise QuorumError(required=required, succeeded=len(record.answers), failures=failures)
    return record


def _model_call(stage: ModelStage) -> ModelCall:
    params = stage.params
    return ModelCall(stage.model, system=params.system, sampling=params.sampling)


def _quorum(stage: ModelStage, *, default: int) -> int:
    return default if stage.params.quorum is None else stage.params.quorum
