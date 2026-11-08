"""What ember knows about a model.

`ModelSpec` records come from one data file, ``catalog.toml``, and nowhere
else. A model that is not in the catalog can still be called; its spec is
marked unknown and has no rates or limits.
"""

import dataclasses
import datetime
import decimal
from typing import Literal

from ember.llm.ref import ModelRef
from ember.llm.request import WireApi
from ember.llm.result import Money, Usage

__all__ = ["LongContextRates", "ModelSpec", "Rates", "WireApi"]

SamplingField = Literal[
    "temperature", "top_p", "max_output_tokens", "stop", "seed", "reasoning_effort"
]

_PER_MILLION = decimal.Decimal(1_000_000)


@dataclasses.dataclass(frozen=True, slots=True)
class LongContextRates:
    """Rates that replace the base rates when a prompt exceeds a threshold.

    Attributes:
        above_prompt_tokens: The rates apply when `Usage.prompt_tokens` is
            greater than this.
        rates: Dollars per million tokens above the threshold.
    """

    above_prompt_tokens: int
    rates: "Rates"


@dataclasses.dataclass(frozen=True, slots=True)
class Rates:
    """Published prices in dollars per million tokens.

    Attributes:
        input: Uncached prompt tokens.
        cache_read: Prompt tokens served from cache.
        cache_write: Prompt tokens written to cache.
        output: Generated tokens, reasoning included.
        source_url: The provider page the prices were read from.
        as_of: The date they were read.
        long_context: Different rates for long prompts, if the provider has them.
    """

    input: decimal.Decimal
    cache_read: decimal.Decimal
    cache_write: decimal.Decimal
    output: decimal.Decimal
    source_url: str
    as_of: datetime.date
    long_context: LongContextRates | None = None

    def cost(self, usage: Usage) -> Money:
        """Prices ``usage`` exactly, rounding once to the nano-dollar.

        Raises:
            ValueError: ``usage`` was not measured, so it cannot be priced.
        """
        if not usage.measured:
            raise ValueError("unmeasured usage cannot be priced")
        rates = self
        if self.long_context and usage.prompt_tokens > self.long_context.above_prompt_tokens:
            rates = self.long_context.rates
        total = (
            usage.input * rates.input
            + usage.cache_read * rates.cache_read
            + usage.cache_write * rates.cache_write
            + usage.output * rates.output
        ) / _PER_MILLION
        return Money.from_dollars(total)


@dataclasses.dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model's identity, limits, accepted settings and prices.

    Attributes:
        ref: The canonical ``provider/name``.
        wire_id: The model id sent to the provider's API.
        api: The wire protocol used to call it.
        aliases: Other names that resolve to this model.
        context_window: Maximum prompt plus output tokens, if known.
        max_output_tokens: Maximum generated tokens, if known.
        accepts: Sampling settings the model accepts; anything else set on a
            request is rejected before sending.
        reasoning_efforts: Allowed values for `Sampling.reasoning_effort`.
        rates: Prices, or ``None`` when unknown.
        known: False for a model resolved without a catalog entry.
    """

    ref: ModelRef
    wire_id: str
    api: WireApi
    aliases: tuple[str, ...] = ()
    context_window: int | None = None
    max_output_tokens: int | None = None
    accepts: frozenset[SamplingField] = frozenset(
        {"temperature", "top_p", "max_output_tokens", "stop", "seed"}
    )
    reasoning_efforts: frozenset[str] = frozenset()
    rates: Rates | None = None
    known: bool = True
