"""A NON plan: exact call counts and labelled output bounds from the catalog."""

from ember import build_graph
from ember.llm.catalog import Catalog
from ember.llm.result import Money

CATALOG = Catalog.from_toml(
    """
[[model]]
id = "openai/priced"
wire_id = "priced"
api = "openai_responses"
aliases = []
max_output_tokens = 1000
accepts = ["max_output_tokens"]
reasoning_efforts = []
sources = ["https://example.com/models/priced"]
as_of = 2026-11-01

[model.rates]
input = "1"
cache_read = "0.1"
output = "10"
source_url = "https://example.com/pricing"
as_of = 2026-11-01

[[model]]
id = "openai/tiered"
wire_id = "tiered"
api = "openai_responses"
aliases = []
max_output_tokens = 100
accepts = ["max_output_tokens"]
reasoning_efforts = []
sources = ["https://example.com/models/tiered"]
as_of = 2026-11-01

[model.rates]
input = "1"
output = "10"
source_url = "https://example.com/pricing"
as_of = 2026-11-01

[model.rates.long_context]
above_prompt_tokens = 200000
input = "2"
output = "15"

[[model]]
id = "openai/unpriced"
wire_id = "unpriced"
api = "openai_responses"
aliases = []
max_output_tokens = 500
accepts = ["max_output_tokens"]
reasoning_efforts = []
sources = ["https://example.com/models/unpriced"]
as_of = 2026-11-01
"""
)


def test_calls_and_bounds_per_stage() -> None:
    graph = build_graph(
        "3E@openai/priced(max_output_tokens=200), 1V@openai/priced, 1J@openai/not-in-catalog"
    )
    ensemble, verifier, judge = graph.plan(CATALOG).stages
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
    plan = build_graph("3E@openai/priced, 1J@openai/not-in-catalog").plan(CATALOG)
    assert (plan.calls, plan.output_tokens_bound, plan.output_cost_bound) == (4, None, None)


def test_a_known_limit_without_a_price_bounds_tokens_but_not_cost() -> None:
    plan = build_graph("2E@openai/unpriced, 1M").plan(CATALOG)
    assert (plan.output_tokens_bound, plan.output_cost_bound) == (1000, None)


def test_a_vote_stage_makes_no_calls_and_totals_add_up() -> None:
    plan = build_graph("2E@openai/priced, 1M").plan(CATALOG)
    assert [stage.calls for stage in plan.stages] == [2, 0]
    assert (plan.calls, plan.output_tokens_bound) == (2, 2000)
    assert plan.output_cost_bound == Money(20_000_000)


def test_the_stage_limit_never_exceeds_the_model_limit() -> None:
    plan = build_graph("1E@openai/priced(max_output_tokens=5000), 1M").plan(CATALOG)
    assert plan.stages[0].max_output_tokens == 1000


def test_cost_bounds_use_the_highest_output_rate_tier() -> None:
    plan = build_graph("2E@openai/tiered, 1M").plan(CATALOG)
    assert plan.output_cost_bound == Money(3_000_000)  # 200 tokens at $15 per million.


def test_verifier_calls_follow_the_answers_it_can_see() -> None:
    graph = build_graph(
        "2E@openai/priced, 1E@openai/priced, 1V@openai/priced, 2J@openai/priced, 1M"
    )
    assert [stage.calls for stage in graph.plan(CATALOG).stages] == [2, 1, 3, 2, 0]
