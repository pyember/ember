"""Static bounds for one run of a NON graph, computed before any request.

A plan counts model calls and bounds output tokens and their cost from each
stage's ``max_output_tokens`` and the model's catalog limit and rates. A model
the catalog does not list has no known limit or price, so its bounds are
``None``. Every figure is an upper bound. Input tokens are not bounded: they
depend on the question and on the answers earlier stages write, and ember
never estimates token counts. Calls the client retries after a failed attempt
are not counted.
"""

import dataclasses
import decimal

from ember.llm.catalog import Catalog
from ember.llm.result import Money
from ember.llm.spec import ModelSpec, Rates
from ember.non.grammar import ModelStage, Role, Stage, VoteStage

__all__ = ["Plan", "StagePlan", "plan"]

_PER_MILLION = decimal.Decimal(1_000_000)


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


def plan(stages: tuple[Stage, ...], catalog: Catalog) -> Plan:
    """Bounds one run of ``stages``.

    Args:
        stages: A parsed graph.
        catalog: Where model limits and rates come from, usually
            ``ember.llm.catalog.default()``.
    """
    planned: list[StagePlan] = []
    answers = 0
    for stage in stages:
        calls = _calls(stage, answers)
        answers = _answers_after(stage, answers)
        if isinstance(stage, ModelStage):
            planned.append(_model_stage_plan(stage, calls, catalog.get(str(stage.model))))
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


def _answers_after(stage: Stage, answers: int) -> int:
    """The most answers that can exist after ``stage``, given ``answers`` before it."""
    if isinstance(stage, VoteStage):
        return 1
    if stage.role is Role.ENSEMBLE:
        return answers + stage.count
    if stage.role is Role.JUDGE:
        return stage.count
    return answers


def _model_stage_plan(stage: ModelStage, calls: int, spec: ModelSpec | None) -> StagePlan:
    model_limit = None if spec is None else spec.max_output_tokens
    limits = [
        limit
        for limit in (stage.params.sampling.max_output_tokens, model_limit)
        if limit is not None
    ]
    if not limits:
        return StagePlan(stage, calls, None, None, None)
    per_call = min(limits)
    tokens = calls * per_call
    rates = None if spec is None else spec.rates
    cost = None if rates is None else _output_cost_bound(rates, tokens)
    return StagePlan(stage, calls, per_call, tokens, cost)


def _output_cost_bound(rates: Rates, tokens: int) -> Money:
    """Prices ``tokens`` output tokens at the highest output rate of any tier."""
    per_million = rates.output
    if rates.long_context is not None:
        per_million = max(per_million, rates.long_context.rates.output)
    return Money.from_dollars(tokens * per_million / _PER_MILLION)
