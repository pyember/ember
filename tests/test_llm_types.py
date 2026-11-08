import datetime
import decimal

import pytest

from ember.llm import Message, ModelRef, Request, Sampling, TextPart, ToolCallPart, ToolSpec, Usage
from ember.llm.result import Money
from ember.llm.spec import LongContextRates, ModelSpec, Rates


def test_model_ref_keeps_everything_after_the_first_slash() -> None:
    parsed = ModelRef.parse("local/meta-llama/Llama-3.1-8B")
    assert parsed == ModelRef("local", "meta-llama/Llama-3.1-8B")
    assert ModelRef.parse("ollama/llama3.1:8b").name == "llama3.1:8b"


@pytest.mark.parametrize("text", ["gpt-5", "/gpt-5", "openai/", " openai/gpt-5"])
def test_model_ref_rejects_malformed_names(text: str) -> None:
    with pytest.raises(ValueError):
        ModelRef.parse(text)


def test_messages_reject_parts_their_role_cannot_hold() -> None:
    with pytest.raises(ValueError, match="cannot hold ToolCallPart"):
        Message("user", (ToolCallPart("c1", "f", {}),))


def test_request_rejects_unknown_tool_choice_and_trailing_assistant_turn() -> None:
    tool = ToolSpec("lookup", "Look something up.", {"type": "object"})
    with pytest.raises(ValueError, match="unknown tool"):
        Request((Message.user("hi"),), tools=(tool,), tool_choice="search")
    with pytest.raises(ValueError, match="last message"):
        Request((Message.user("hi"), Message.assistant("hello")))
    assert Request((Message("user", (TextPart("hi"),)),), tools=(tool,), tool_choice="lookup")


def test_sampling_validates_ranges() -> None:
    with pytest.raises(ValueError):
        Sampling(temperature=3)
    with pytest.raises(ValueError):
        Sampling(max_output_tokens=0)


def test_usage_invariants() -> None:
    with pytest.raises(ValueError):
        Usage(output=1, reasoning=2)
    with pytest.raises(ValueError):
        Usage(input=1, measured=False)
    total = Usage(input=1, output=2) + Usage(cache_read=3, output=1, reasoning=1)
    assert total == Usage(input=1, cache_read=3, output=3, reasoning=1)
    assert not (Usage() + Usage(measured=False)).measured


def _rates(**overrides: object) -> Rates:
    fields: dict[str, object] = {
        "input": decimal.Decimal("0.15"),
        "cache_read": decimal.Decimal("0.075"),
        "cache_write": decimal.Decimal("0"),
        "output": decimal.Decimal("0.60"),
        "source_url": "https://example.com/pricing",
        "as_of": datetime.date(2026, 11, 7),
    }
    fields.update(overrides)
    return Rates(**fields)  # type: ignore[arg-type]


def test_cost_is_exact_for_small_calls() -> None:
    # 3 input tokens at $0.15 per million is 450 nano-dollars, not zero.
    assert _rates().cost(Usage(input=3)) == Money(450)
    assert _rates().cost(Usage(input=1, cache_read=1, output=1)) == Money(150 + 75 + 600)


def test_long_context_rates_apply_above_the_threshold() -> None:
    expensive = _rates(input=decimal.Decimal("0.30"))
    rates = _rates(long_context=LongContextRates(above_prompt_tokens=10, rates=expensive))
    assert rates.cost(Usage(input=10)) == Money(1500)
    assert rates.cost(Usage(input=11)) == Money(3300)


def test_unmeasured_usage_cannot_be_priced() -> None:
    with pytest.raises(ValueError):
        _rates().cost(Usage(measured=False))


def test_money_formats_exactly_with_at_least_cents() -> None:
    assert str(Money(1_500_000_000)) == "$1.50"
    assert str(Money(450)) == "$0.00000045"
    assert str(Money(0)) == "$0.00"


def test_a_missing_cache_price_leaves_only_calls_that_use_the_cache_unpriced() -> None:
    rates = _rates(cache_read=None, cache_write=None)
    assert not rates.fully_priced
    assert rates.cost(Usage(input=3, output=1)) == Money(450 + 600)
    assert rates.cost(Usage(input=3, cache_read=1)) is None
    assert rates.cost(Usage(input=3, cache_write=1)) is None
    assert _rates().fully_priced


def test_long_context_rates_without_a_cache_price_are_not_fully_priced() -> None:
    long = LongContextRates(above_prompt_tokens=10, rates=_rates(cache_write=None))
    assert not _rates(long_context=long).fully_priced


def test_model_spec_offers_reasoning_effort_exactly_when_it_lists_efforts() -> None:
    ref = ModelRef("openai", "m")
    common = {"ref": ref, "wire_id": "m", "api": "openai_responses", "known": False}
    with pytest.raises(ValueError, match="reasoning_effort"):
        ModelSpec(**common, accepts=frozenset({"reasoning_effort"}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reasoning_effort"):
        ModelSpec(**common, accepts=frozenset(), reasoning_efforts=frozenset({"low"}))  # type: ignore[arg-type]


def test_a_catalog_model_must_say_where_its_values_came_from() -> None:
    with pytest.raises(ValueError, match="sources and as_of"):
        ModelSpec(ref=ModelRef("openai", "m"), wire_id="m", api="openai_responses")
