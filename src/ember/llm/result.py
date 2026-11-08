"""What a model call produced.

Every field of a `Result` is required: a call that cannot fill one fails
instead of guessing. Token counts come from the provider and are never
estimated; money is an integer number of nano-dollars.
"""

import dataclasses
import decimal
import enum
from collections.abc import Mapping
from typing import TypeAlias

from ember.llm.ref import ModelRef
from ember.llm.request import JsonValue, Message

__all__ = [
    "Done",
    "Event",
    "FinishReason",
    "Money",
    "ReasoningDelta",
    "Result",
    "TextDelta",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
    "UsageEvent",
]

_NANOS_PER_DOLLAR = 1_000_000_000


class FinishReason(enum.Enum):
    """Why generation ended."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    REFUSAL = "refusal"


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class Money:
    """An amount of US dollars, held exactly as integer nano-dollars."""

    nanos: int

    @classmethod
    def from_dollars(cls, dollars: decimal.Decimal) -> "Money":
        """Converts exactly, rounding half to even only below one nano-dollar."""
        nanos = (dollars * _NANOS_PER_DOLLAR).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)
        return cls(int(nanos))

    @property
    def dollars(self) -> decimal.Decimal:
        """The amount in dollars, exactly."""
        return decimal.Decimal(self.nanos) / _NANOS_PER_DOLLAR

    def __add__(self, other: "Money") -> "Money":
        return Money(self.nanos + other.nanos)

    def __str__(self) -> str:
        whole, _, fraction = f"{self.dollars:f}".partition(".")
        return f"${whole}.{fraction.ljust(2, '0')}"


@dataclasses.dataclass(frozen=True, slots=True)
class Usage:
    """Tokens a call consumed, split the way providers bill them.

    Attributes:
        input: Prompt tokens read without a cache hit.
        cache_read: Prompt tokens served from the provider's prompt cache.
        cache_write: Prompt tokens written to the provider's prompt cache.
        output: Generated tokens, reasoning included.
        reasoning: The part of `output` spent on reasoning.
        measured: False when the provider reported no counts; every count is
            then zero and must not be read as "free".
    """

    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    reasoning: int = 0
    measured: bool = True

    def __post_init__(self) -> None:
        counts = (self.input, self.cache_read, self.cache_write, self.output, self.reasoning)
        if min(counts) < 0:
            raise ValueError("token counts cannot be negative")
        if self.reasoning > self.output:
            raise ValueError("reasoning tokens are part of output and cannot exceed it")
        if not self.measured and any(counts):
            raise ValueError("unmeasured usage carries no counts")

    @property
    def prompt_tokens(self) -> int:
        """All prompt tokens: uncached, cache reads and cache writes."""
        return self.input + self.cache_read + self.cache_write

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input=self.input + other.input,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
            measured=self.measured and other.measured,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call the model asked for."""

    id: str
    name: str
    arguments: Mapping[str, JsonValue]


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """The complete outcome of one successful model call.

    Attributes:
        text: Generated text; empty when the model only called tools.
        reasoning: Reasoning summary text, when the provider returns one.
        tool_calls: Tool calls in the order the model made them.
        finish_reason: Why generation ended.
        usage: Tokens consumed.
        cost: Cost of the call, or ``None`` when the model's price is unknown.
        model: The model that served the call.
        request_id: The provider's id for the request, when it sent one.
        latency_s: Seconds from sending the request to the last byte.
        message: The assistant turn to append to the conversation to continue
            it: text, tool calls and opaque reasoning state in the order the
            provider produced them. ``None`` when the model produced nothing.
    """

    text: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    cost: Money | None
    model: ModelRef
    request_id: str | None
    latency_s: float
    message: Message | None


@dataclasses.dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of generated text."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """A piece of reasoning summary text."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A piece of a tool call; `index` identifies the call within the response."""

    index: int
    id: str | None
    name: str | None
    arguments_json: str


@dataclasses.dataclass(frozen=True, slots=True)
class UsageEvent:
    """Token usage reported during a stream."""

    usage: Usage


@dataclasses.dataclass(frozen=True, slots=True)
class Done:
    """The last event of every successful stream."""

    result: Result


Event: TypeAlias = TextDelta | ReasoningDelta | ToolCallDelta | UsageEvent | Done
