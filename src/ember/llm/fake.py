"""A provider for tests and examples that answers from code. It never touches the network.

`FakeProvider` answers each request with a `Reply` (text, tool calls, usage)
or a `Fail` (a `ModelError` of a chosen kind). It decides which one in one of
three ways:

- ``respond(model, request)``: a function of the call, for tests whose calls
  run concurrently or must answer differently per model or prompt;
- ``script``: answers given in order, across all requests;
- ``default``: the answer when neither applies.

It records every request it receives, in arrival order, and the settings of
every provider a client built from it. Install it behind `ember.models` with
``ember.llm.using(Client(provider_factory=fake.factory))``.

Example:
    >>> import ember
    >>> from ember.errors import ErrorKind
    >>> from ember.llm import Client, ModelRef, using
    >>> from ember.llm.fake import Fail, FakeProvider, Reply
    >>> def respond(model, request):
    ...     if model == ModelRef("anthropic", "claude-haiku-5-5"):
    ...         return Fail(ErrorKind.AUTH, status=401)
    ...     return Reply(f"{model.name} says hi")
    >>> fake = FakeProvider(respond=respond)
    >>> with Client(provider_factory=fake.factory) as client, using(client):
    ...     print(ember.models("gpt-5.5", "Hello?").text)
    ...     ember.models("claude-haiku-5-5", "Hello?")
    Traceback (most recent call last):
    ...
    ember.errors.ModelError: auth: scripted failure model=anthropic/claude-haiku-5-5 status=401
    >>> [seen.model.name for seen in fake.requests]
    ['gpt-5.5', 'claude-haiku-5-5']
"""

import collections
import dataclasses
import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import TypeAlias

from ember.context import Context, ProviderSettings
from ember.errors import ErrorKind, ModelError
from ember.llm.provider import Decoded, Provider, StreamItem
from ember.llm.ref import ModelRef
from ember.llm.request import (
    Message,
    Part,
    ReasoningPart,
    Request,
    TextPart,
    ToolCallPart,
)
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

__all__ = ["Answer", "Fail", "FakeProvider", "Reply", "Seen"]


@dataclasses.dataclass(frozen=True, slots=True)
class Reply:
    """A successful answer.

    Attributes:
        text: Generated text.
        reasoning: Reasoning summary text.
        tool_calls: Tool calls to report.
        finish_reason: Why generation ended.
        usage: Token usage to report; unmeasured by default.
        request_id: The provider request id to report.
        cost: A provider-reported cost, if the fake should report one.
        latency_s: Seconds to sleep before answering, on the thread that
            runs the request.
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
    latency_s: float = 0.0
    stream_error: ErrorKind | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Fail:
    """A failed answer: the fake raises `ModelError` with these fields.

    Attributes:
        kind: Why the call fails.
        status: The HTTP status to report.
        retry_after_s: The wait the provider asks for.
        latency_s: Seconds to sleep before failing.
    """

    kind: ErrorKind
    status: int | None = None
    retry_after_s: float | None = None
    latency_s: float = 0.0


Answer: TypeAlias = Reply | Fail
Respond: TypeAlias = Callable[[ModelRef, Request], Answer]


@dataclasses.dataclass(frozen=True, slots=True)
class Seen:
    """A request the fake received.

    Attributes:
        model: The model the request was for, as the catalog names it.
        wire_id: The model id a real provider would have sent.
        request: The request itself.
        streamed: Whether it was a streaming request.
    """

    model: ModelRef
    wire_id: str
    request: Request
    streamed: bool


_OK = Reply()


class FakeProvider:
    """A provider that answers from code. Thread-safe.

    Pass `factory` as a client's ``provider_factory``; every provider the
    client needs is then this fake.
    """

    def __init__(
        self,
        script: Iterable[Answer] = (),
        *,
        respond: Respond | None = None,
        default: Answer | None = _OK,
    ) -> None:
        """Creates a fake.

        Args:
            script: Answers to give, in order, across all requests.
            respond: Chooses each answer from the model and request. Cannot be
                combined with ``script``.
            default: The answer once the script is used up; ``None`` makes a
                request past the end of the script fail the test loudly.

        Raises:
            ValueError: Both ``script`` and ``respond`` were given.
        """
        self._script = collections.deque(script)
        if respond is not None and self._script:
            raise ValueError("give FakeProvider a script or a respond function, not both")
        self._respond = respond
        self._default = default
        self._lock = threading.Lock()
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
        """Returns the chosen reply, or raises the chosen failure."""
        return _decoded(spec, self._answer(spec, request, streamed=False))

    def stream(self, spec: ModelSpec, request: Request) -> Iterator[StreamItem]:
        """Streams the chosen reply piece by piece, ending with `Decoded`."""
        reply = self._answer(spec, request, streamed=True)
        if reply.reasoning:
            yield ReasoningDelta(reply.reasoning)
        for piece in re.findall(r"\S+\s*|\s+", reply.text):
            yield TextDelta(piece)
        if reply.stream_error is not None:
            raise _error(spec, Fail(reply.stream_error), before_first_byte=False)
        for index, call in enumerate(reply.tool_calls):
            yield ToolCallDelta(index, call.id, call.name, json.dumps(call.arguments))
        if reply.usage.measured:
            yield UsageEvent(reply.usage)
        yield _decoded(spec, reply)

    def close(self) -> None:
        """Counts the close; the fake holds no connections."""
        with self._lock:
            self._closes += 1

    def _answer(self, spec: ModelSpec, request: Request, *, streamed: bool) -> Reply:
        with self._lock:
            self._seen.append(Seen(spec.ref, spec.wire_id, request, streamed))
            scripted = self._script.popleft() if self._script else None
        if scripted is not None:
            answer: Answer | None = scripted
        elif self._respond is not None:
            answer = self._respond(spec.ref, request)
        else:
            answer = self._default
        if answer is None:
            raise AssertionError("FakeProvider script is used up and has no default")
        if answer.latency_s:
            time.sleep(answer.latency_s)
        if isinstance(answer, Fail):
            raise _error(spec, answer, before_first_byte=True)
        return answer


def _decoded(spec: ModelSpec, reply: Reply) -> Decoded:
    return Decoded(
        text=reply.text,
        reasoning=reply.reasoning,
        tool_calls=reply.tool_calls,
        finish_reason=reply.finish_reason,
        usage=reply.usage,
        request_id=reply.request_id,
        message=_assistant_turn(spec, reply),
        cost=reply.cost,
    )


def _assistant_turn(spec: ModelSpec, reply: Reply) -> Message:
    """The reply as the assistant message a provider would return: reasoning,
    text, then tool calls, or one empty text part when there is nothing."""
    parts: list[Part] = []
    if reply.reasoning:
        parts.append(ReasoningPart(reply.reasoning, spec.api, spec.ref.provider, {}))
    if reply.text:
        parts.append(TextPart(reply.text))
    parts.extend(ToolCallPart(call.id, call.name, call.arguments) for call in reply.tool_calls)
    return Message("assistant", tuple(parts) or (TextPart(""),))


def _error(spec: ModelSpec, fail: Fail, *, before_first_byte: bool) -> ModelError:
    return ModelError(
        fail.kind,
        provider=spec.ref.provider,
        model=spec.ref.name,
        reason="scripted failure",
        status=fail.status,
        retry_after_s=fail.retry_after_s,
        before_first_byte=before_first_byte,
    )
