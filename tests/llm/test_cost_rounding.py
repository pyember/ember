"""Exact pricing can round the last nano-dollar up, for upper bounds."""

import datetime
import decimal

from ember.llm.result import Money, Usage
from ember.llm.spec import Rates


def _rates(output: str) -> Rates:
    return Rates(
        input=decimal.Decimal("1"),
        cache_read=None,
        cache_write=None,
        output=decimal.Decimal(output),
        source_url="https://example.com/pricing",
        as_of=datetime.date(2026, 11, 1),
    )


def test_money_rounds_half_to_even_by_default_and_up_on_request() -> None:
    dollars = decimal.Decimal("0.0000000014")  # 1.4 nano-dollars.
    assert Money.from_dollars(dollars) == Money(1)
    assert Money.from_dollars(dollars, rounding=decimal.ROUND_CEILING) == Money(2)


def test_cost_rounds_up_on_request_and_is_otherwise_unchanged() -> None:
    rates = _rates("0.0000015")  # One output token costs 0.0015 nano-dollars.
    assert rates.cost(Usage(output=1)) == Money(0)
    assert rates.cost(Usage(output=1), rounding=decimal.ROUND_CEILING) == Money(1)
    assert _rates("15").cost(Usage(output=1), rounding=decimal.ROUND_CEILING) == Money(15_000)
