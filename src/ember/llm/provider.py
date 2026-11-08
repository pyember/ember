"""The provider interface.

A provider turns a `Request` into a provider's HTTP traffic and back. It is a
pure codec for one wire format plus a transport that owns the HTTP client and
authentication. Providers never retry and never price; the client does both.
"""

import dataclasses
from collections.abc import Iterator
from typing import Protocol, TypeAlias

from ember.llm.request import Message, Request
from ember.llm.result import (
    FinishReason,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageEvent,
)
from ember.llm.spec import ModelSpec

__all__ = ["Decoded", "Provider", "StreamItem"]


@dataclasses.dataclass(frozen=True, slots=True)
class Decoded:
    """What a provider's response says, before the client adds cost and timing.

    Attributes:
        text: Generated text.
        reasoning: Reasoning summary text, if any.
        tool_calls: Tool calls in order.
        finish_reason: Why generation ended, mapped from the provider's value.
        usage: Token usage as the provider reported it.
        request_id: The provider's id for the request.
        message: The assistant turn as the provider produced it, reasoning
            state included; a single empty `TextPart` when it produced
            nothing. `text`, `reasoning` and `tool_calls` are derived from it.
    """

    text: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    request_id: str | None
    message: Message


StreamItem: TypeAlias = TextDelta | ReasoningDelta | ToolCallDelta | UsageEvent | Decoded
"""A streamed piece; a successful stream ends with exactly one `Decoded`."""


class Provider(Protocol):
    """Sends requests to one configured provider.

    Implementations raise `ember.errors.ModelError` for every failure, with
    the kind mapped from the HTTP status or the provider's error event. They
    never embed upstream response bodies in error messages.
    """

    @property
    def name(self) -> str:
        """The provider name used in model ids."""
        ...

    def send(self, spec: ModelSpec, request: Request) -> Decoded:
        """Sends ``request`` and returns the decoded response."""
        ...

    def stream(self, spec: ModelSpec, request: Request) -> Iterator[StreamItem]:
        """Sends ``request`` with streaming and yields pieces as they arrive."""
        ...

    def close(self) -> None:
        """Releases the provider's HTTP connections."""
        ...
