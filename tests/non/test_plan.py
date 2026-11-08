"""A NON plan: exact call counts and labelled output bounds from the catalog."""

import datetime
import decimal

from ember import build_graph
from ember.llm.ref import ModelRef
from ember.llm.result import Money
from ember.llm.spec import LongContextRates, ModelSpec, Rates


def _rates(output: str, long_output: str | None = None) -> Rates:
    long_context = None
    if long_output is not None:
        long_context = LongContextRates(200_000, _rates(long_output))
    return Rates(
        input=decimal.Decimal("1"),
        cache_read=decimal.Decimal("0.1"),
        cache_write=decimal.Decimal("1.25"),
        output=decimal.Decimal(output),
        source_url="https://example.com/pricing",
        as_of=datetime.date(2026, 11, 1),
        long_context=long_context,
    )


SPECS = {
    "a/priced": ModelSpec(
        ModelRef("a", "priced"), "priced", "openai_chat", max_output_tokens=1000, rates=_rates("10")
    ),
    "a/tiered": ModelSpec(
        ModelRef("a", "tiered"),
        "tiered",
        "openai_chat",
        max_output_tokens=100,
        rates=_rates("10", long_output="15"),
    ),
    "a/unknown": ModelSpec(ModelRef("a", "unknown"), "unknown", "openai_chat", known=False),
}


def _resolve(model: ModelRef) -> ModelSpec:
    return SPECS[str(model)]


def test_calls_and_bounds_per_stage() -> None:
    graph = build_graph("3E@a/priced(max_output_tokens=200), 1V@a/priced, 1J@a/unknown")
    ensemble, verifier, judge = graph.plan(_resolve).stages
    assert (ensemble.calls, ensemble.max_output_tokens, ensemble.output_tokens_bound) == (
        3,
        200,
        600,
    )
    assert ensemble.output_cost_bound == Money(6_000_000)
    assert (verifier.calls, verifier.max_output_tokens, verifier.output_tokens_bound) == (
        3,
        1000,
        3000,
    )
    assert verifier.output_cost_bound == Money(30_000_000)
    assert (judge.calls, judge.max_output_tokens, judge.output_cost_bound) == (1, None, None)


def test_totals_are_unknown_when_any_stage_is_unbounded() -> None:
    plan = build_graph("3E@a/priced, 1J@a/unknown").plan(_resolve)
    assert (plan.calls, plan.output_tokens_bound, plan.output_cost_bound) == (4, None, None)


def test_a_vote_stage_makes_no_calls_and_totals_add_up() -> None:
    plan = build_graph("2E@a/priced, 1M").plan(_resolve)
    assert [stage.calls for stage in plan.stages] == [2, 0]
    assert (plan.calls, plan.output_tokens_bound) == (2, 2000)
    assert plan.output_cost_bound == Money(20_000_000)


def test_the_stage_limit_never_exceeds_the_model_limit() -> None:
    plan = build_graph("1E@a/priced(max_output_tokens=5000), 1M").plan(_resolve)
    assert plan.stages[0].max_output_tokens == 1000


def test_cost_bounds_use_the_highest_output_rate_tier() -> None:
    plan = build_graph("2E@a/tiered, 1M").plan(_resolve)
    assert plan.output_cost_bound == Money(3_000_000)  # 200 tokens at $15 per million.


def test_verifier_calls_follow_the_answers_it_can_see() -> None:
    plan = build_graph("2E@a/priced, 1E@a/priced, 2J@a/priced, 1V@a/priced, 1J@a/priced")
    assert [stage.calls for stage in plan.plan(_resolve).stages] == [2, 1, 2, 2, 1]
