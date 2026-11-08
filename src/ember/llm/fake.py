"""A scripted provider for tests and examples. It never touches the network.

`FakeProvider` answers each request with the next item of its script: a
`Reply` to return, or an `ErrorKind` or `ModelError` to raise. When the
script runs out it answers with ``default``. It records every request it sees
and every provider configuration it was built for.

Example:
    >>> import ember
    >>> from ember.errors import ErrorKind
    >>> from ember.llm import Client, using
    >>> from ember.llm.fake import FakeProvider, Reply
    >>> fake = FakeProvider([ErrorKind.OVERLOADED, Reply("Paris")])
    >>> with Client(provider_factory=fake.factory) as client, using(client):
    ...     print(ember.models("openai/gpt-5.5", "Capital of France?").text)
    Paris
    >>> len(fake.requests)  # the overloaded attempt was retried
    2
"""

import collections
import dataclasses
import json
import re
import threading
import time
from collections.abc import Iterable, Iterator

from ember.context import Context, ProviderSettings
from ember.errors import ErrorKind, ModelError
from ember.llm.provider import Decoded, Provider, StreamItem
from ember.llm.ref import ModelRef
from ember.llm.request import Request
from ember.llm.result import (
    FinishReason,
    Money,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageEvent,
)
from ember.llm.spec import ModelSpec

__all__ = ["FakeProvider", "Reply", "Seen"]


@dataclasses.dataclass(frozen=True, slots=True)
class Reply:
    """One scripted answer.

    Attributes:
        text: Generated text.
        reasoning: Reasoning summary text.
        tool_calls: Tool calls to report.
        finish_reason: Why generation ended.
        usage: Token usage to report; unmeasured by default.
        request_id: The provider request id to report.
        cost: A provider-reported cost, if the fake should report one.
        stream_error: When streaming, raise this kind after the text has been
            delivered, as a connection dropped mid-stream would.
    """

    text: str = "ok"
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Usage(measured=False)
    request_id: str | None = None
    cost: Money | None = None
    stream_error: ErrorKind | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Seen:
    """A request the fake received.

    Attributes:
        model: The model the request was for.
        wire_id: The model id a real provider would have sent.
        request: The request itself.
        streamed: Whether it was a streaming request.
    """

    model: ModelRef
    wire_id: str
    request: Request
    streamed: bool


ScriptItem = Reply | ErrorKind | ModelError

_OK = Reply()


class FakeProvider:
    """A provider that answers from a script. Thread-safe.

    Pass `factory` as a client's ``provider_factory``; every provider the
    client needs is then this fake.
    """

    def __init__(
        self,
        script: Iterable[ScriptItem] = (),
        *,
        default: Reply | None = _OK,
        latency_s: float = 0.0,
    ) -> None:
        """Creates a fake.

        Args:
            script: Answers to give, in order, across all requests.
            default: The answer once the script is used up; ``None`` makes a
                request past the end of the script fail the test loudly.
            latency_s: Seconds each request takes.
        """
        self._lock = threading.Lock()
        self._script = collections.deque(script)
        self._default = default
        self._latency_s = latency_s
        self._seen: list[Seen] = []
        self._built: list[ProviderSettings] = []
        self._closes = 0

    @property
    def name(self) -> str:
        """The provider name; the fake stands in for every provider."""
        return "fake"

    @property
    def requests(self) -> tuple[Seen, ...]:
        """Every request received so far, in arrival order."""
        with self._lock:
            return tuple(self._seen)

    @property
    def built(self) -> tuple[ProviderSettings, ...]:
        """The settings of every provider a client built from this fake."""
        with self._lock:
            return tuple(self._built)

    @property
    def closes(self) -> int:
        """How many times a client closed this fake."""
        with self._lock:
            return self._closes

    def factory(self, settings: ProviderSettings, ctx: Context) -> Provider:
        """A provider factory that records ``settings`` and returns this fake."""
        del ctx
        with self._lock:
            self._built.append(settings)
        return self

    def send(self, spec: ModelSpec, request: Request) -> Decoded:
        """Returns the next scripted reply, or raises the next scripted error."""
        reply = self._next(spec, request, streamed=False)
        return _decoded(reply)

    def stream(self, spec: ModelSpec, request: Request) -> Iterator[StreamItem]:
        """Streams the next scripted reply piece by piece, ending with `Decoded`."""
        reply = self._next(spec, request, streamed=True)
        if reply.reasoning:
            yield ReasoningDelta(reply.reasoning)
        for piece in re.findall(r"\S+\s*|\s+", reply.text):
            yield TextDelta(piece)
        if reply.stream_error is not None:
            raise _error(reply.stream_error, spec, before_first_byte=False)
        for index, call in enumerate(reply.tool_calls):
            yield ToolCallDelta(index, call.id, call.name, json.dumps(call.arguments))
        if reply.usage.measured:
            yield UsageEvent(reply.usage)
        yield _decoded(reply)

    def close(self) -> None:
        """Counts the close; the fake holds no connections."""
        with self._lock:
            self._closes += 1

    def _next(self, spec: ModelSpec, request: Request, *, streamed: bool) -> Reply:
        with self._lock:
            self._seen.append(Seen(spec.ref, spec.wire_id, request, streamed))
            item = self._script.popleft() if self._script else self._default
        if self._latency_s:
            time.sleep(self._latency_s)
        if item is None:
            raise AssertionError("FakeProvider script is used up and has no default")
        if isinstance(item, ErrorKind):
            raise _error(item, spec, before_first_byte=True)
        if isinstance(item, ModelError):
            raise item
        return item


def _decoded(reply: Reply) -> Decoded:
    return Decoded(
        text=reply.text,
        reasoning=reply.reasoning,
        tool_calls=reply.tool_calls,
        finish_reason=reply.finish_reason,
        usage=reply.usage,
        request_id=reply.request_id,
        cost=reply.cost,
    )


def _error(kind: ErrorKind, spec: ModelSpec, *, before_first_byte: bool) -> ModelError:
    return ModelError(
        kind,
        provider=spec.ref.provider,
        model=spec.ref.name,
        reason="scripted failure",
        before_first_byte=before_first_byte,
    )
