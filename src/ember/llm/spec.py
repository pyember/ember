"""What ember knows about a model.

`ModelSpec` records come from one data file, ``catalog.toml``, and nowhere
else. A model that is not in the catalog can still be called; its spec is
marked unknown and has no rates, limits or accepted-settings list.
"""

import dataclasses
import datetime
import decimal
from typing import Literal

from ember.llm.ref import ModelRef
from ember.llm.request import WireApi
from ember.llm.result import Money, Usage

__all__ = ["SAMPLING_FIELDS", "LongContextRates", "ModelSpec", "Rates", "SamplingField", "WireApi"]

SamplingField = Literal[
    "temperature", "top_p", "max_output_tokens", "stop", "seed", "reasoning_effort"
]
SAMPLING_FIELDS: frozenset[SamplingField] = frozenset(
    {"temperature", "top_p", "max_output_tokens", "stop", "seed", "reasoning_effort"}
)

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
        cache_read: Prompt tokens served from cache, or ``None`` when the
            provider publishes no such price for the model.
        cache_write: Prompt tokens written to cache, or ``None`` when the
            provider publishes no such price for the model.
        output: Generated tokens, reasoning included.
        source_url: The provider page the prices were read from.
        as_of: The date they were read.
        long_context: Different rates for long prompts, if the provider has them.
    """

    input: decimal.Decimal
    cache_read: decimal.Decimal | None
    cache_write: decimal.Decimal | None
    output: decimal.Decimal
    source_url: str
    as_of: datetime.date
    long_context: LongContextRates | None = None

    @property
    def fully_priced(self) -> bool:
        """Whether every kind of token has a published price, at every prompt size."""
        own = self.cache_read is not None and self.cache_write is not None
        return own and (self.long_context is None or self.long_context.rates.fully_priced)

    def cost(self, usage: Usage) -> Money | None:
        """Prices ``usage`` exactly, rounding once to the nano-dollar.

        Returns:
            The cost, or ``None`` when ``usage`` includes cache reads or writes
            and the model has no published price for them.

        Raises:
            ValueError: ``usage`` was not measured, so it cannot be priced.
        """
        if not usage.measured:
            raise ValueError("unmeasured usage cannot be priced")
        rates = self
        if self.long_context and usage.prompt_tokens > self.long_context.above_prompt_tokens:
            rates = self.long_context.rates
        cache_read = _priced(usage.cache_read, rates.cache_read)
        cache_write = _priced(usage.cache_write, rates.cache_write)
        if cache_read is None or cache_write is None:
            return None
        total = usage.input * rates.input + cache_read + cache_write + usage.output * rates.output
        return Money.from_dollars(total / _PER_MILLION)


def _priced(tokens: int, rate: decimal.Decimal | None) -> decimal.Decimal | None:
    """Tokens times rate; zero tokens cost nothing even without a rate."""
    if tokens == 0:
        return decimal.Decimal(0)
    return None if rate is None else tokens * rate


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
            Non-empty exactly when `accepts` contains ``"reasoning_effort"``.
        rates: Prices, or ``None`` when unknown.
        known: False for a model resolved without a catalog entry. Nothing is
            known about such a model's limits or settings, so none are checked
            before sending; the provider rejects what it does not accept.
        sources: The provider pages the identity, limits and settings were
            read from.
        as_of: The date they were read; ``None`` for an unknown model.
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
    sources: tuple[str, ...] = ()
    as_of: datetime.date | None = None

    def __post_init__(self) -> None:
        if ("reasoning_effort" in self.accepts) != bool(self.reasoning_efforts):
            raise ValueError(
                f"{self.ref}: accepts must include reasoning_effort exactly when "
                "reasoning_efforts is non-empty"
            )
        if self.known and (not self.sources or self.as_of is None):
            raise ValueError(f"{self.ref}: a catalog model needs sources and as_of")
