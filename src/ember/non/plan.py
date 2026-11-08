"""Static bounds for one run of a NON graph, computed before any request.

A plan counts model calls and bounds output tokens and their cost from each
stage's ``max_output_tokens`` or, failing that, the model's catalog limit and
rates. Every figure is an upper bound. Input tokens are not bounded: they
depend on the question and on the answers earlier stages write, and ember
never estimates token counts. Calls the client retries after a failed attempt
are not counted.
"""

import dataclasses
from collections.abc import Callable

from ember.llm.ref import ModelRef
from ember.llm.result import Money, Usage
from ember.llm.spec import ModelSpec, Rates
from ember.non.grammar import ModelStage, Role, Stage, VoteStage, max_answers_after

__all__ = ["Plan", "StagePlan", "plan"]


@dataclasses.dataclass(frozen=True, slots=True)
class StagePlan:
    """Upper bounds for one stage.

    Attributes:
        stage: The parsed stage.
        calls: The most model calls the stage makes.
        max_output_tokens: The most output tokens one call can produce, or
            ``None`` when neither the stage nor the catalog sets a limit.
        output_tokens_bound: The most output tokens the stage can produce, or
            ``None`` when unbounded.
        output_cost_bound: The most those tokens can cost, or ``None`` when
            they are unbounded or the model's price is unknown.
    """

    stage: Stage
    calls: int
    max_output_tokens: int | None
    output_tokens_bound: int | None
    output_cost_bound: Money | None


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """Upper bounds for one run of a graph, stage by stage.

    Attributes:
        stages: One entry per stage, in specification order.
    """

    stages: tuple[StagePlan, ...]

    @property
    def calls(self) -> int:
        """The most model calls one run makes."""
        return sum(stage.calls for stage in self.stages)

    @property
    def output_tokens_bound(self) -> int | None:
        """The most output tokens one run produces, or ``None`` when unbounded."""
        bounds = [stage.output_tokens_bound for stage in self.stages]
        return None if None in bounds else sum(bound for bound in bounds if bound is not None)

    @property
    def output_cost_bound(self) -> Money | None:
        """The most one run's output tokens cost, or ``None`` when unknown."""
        total = Money(0)
        for stage in self.stages:
            if stage.output_cost_bound is None:
                return None
            total += stage.output_cost_bound
        return total


def plan(stages: tuple[Stage, ...], resolve: Callable[[ModelRef], ModelSpec]) -> Plan:
    """Bounds one run of ``stages``.

    Args:
        stages: A parsed graph.
        resolve: Returns the catalog spec for a model, such as the catalog's
            resolution function.
    """
    planned: list[StagePlan] = []
    answers = 0
    for stage in stages:
        calls = _calls(stage, answers)
        answers = max_answers_after(stage, answers)
        if isinstance(stage, ModelStage):
            planned.append(_model_stage_plan(stage, calls, resolve(stage.model)))
        else:
            planned.append(StagePlan(stage, 0, None, 0, Money(0)))
    return Plan(tuple(planned))


def _calls(stage: Stage, answers_before: int) -> int:
    """E and J stages make ``count`` calls, V one per answer it sees, M none."""
    if isinstance(stage, VoteStage):
        return 0
    if stage.role is Role.VERIFIER:
        return answers_before
    return stage.count


def _model_stage_plan(stage: ModelStage, calls: int, spec: ModelSpec) -> StagePlan:
    limits = [
        limit
        for limit in (stage.params.sampling.max_output_tokens, spec.max_output_tokens)
        if limit is not None
    ]
    if not limits:
        return StagePlan(stage, calls, None, None, None)
    per_call = min(limits)
    tokens = calls * per_call
    cost = None if spec.rates is None else _output_cost_bound(spec.rates, tokens)
    return StagePlan(stage, calls, per_call, tokens, cost)


def _output_cost_bound(rates: Rates, tokens: int) -> Money:
    usage = Usage(output=tokens)
    cost = rates.cost(usage)
    if rates.long_context is not None:
        cost = max(cost, rates.long_context.rates.cost(usage))
    return cost
