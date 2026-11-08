"""The client: the one path every model call takes.

`Client.send` runs, in order:

1. Resolve the model against the catalog (`ModelError(NOT_FOUND)` for an
   unknown bare name).
2. Check the request's settings against what the model accepts
   (`ValueError`).
3. Apply the price policy: under ``require_price``, a model without a full
   published price raises `ModelError(UNPRICED)`.
4. Lease the provider built from the *active* context's settings, so
   ``ember.context.override(providers=...)`` always takes effect.
5. Wait for a rate-limit permit through `ember.xcs.runtime.wait`.
6. Hand the call to `ember.xcs.runtime.dispatch`; `ember.llm.attempt` sends,
   retries, prices and reports it.

Steps 1-5 raise at the call site. Where a failure from step 6 raises is
decided by `ember.xcs.runtime.deferred()` alone: at the call site outside
``@xcs.jit`` and ``xcs.map``, at the first read of the result inside them.

`using(client)` makes a client the target of `ember.models` for a block;
otherwise `ember.models` uses one process-wide default client, built on first
use, closed at interpreter exit and discarded in a forked child.
"""

import atexit
import contextlib
import contextvars
import dataclasses
import os
import random
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from types import MappingProxyType, TracebackType

from ember import context
from ember.context import Context, ProviderSettings
from ember.errors import ConfigError, ErrorKind, ModelError
from ember.llm import catalog as catalog_module
from ember.llm import observe, providers
from ember.llm.attempt import Call, extensions_keep_list_price
from ember.llm.catalog import Catalog
from ember.llm.completion import Completion
from ember.llm.limiter import Limiter, LimiterState, Permit
from ember.llm.pool import ProviderFactory, ProviderPool
from ember.llm.ref import ModelRef
from ember.llm.request import (
    DEFAULT_SAMPLING,
    JsonValue,
    Message,
    Request,
    Sampling,
    ToolChoice,
    ToolSpec,
)
from ember.llm.result import Event
from ember.llm.spec import ModelSpec, SamplingField
from ember.xcs import runtime

__all__ = [
    "BoundModel",
    "CallOptions",
    "Client",
    "active",
    "check_settings",
    "using",
]

ModelName = str | ModelRef | None
"""``provider/name``, a catalog alias, a `ModelRef`, or ``None`` for the default."""

Prompt = str | Sequence[Message]
"""A user message's text, or a whole conversation."""


@dataclasses.dataclass(frozen=True, slots=True)
class CallOptions:
    """Everything about a call except the model and the prompt.

    The fields mirror the keyword arguments of `ember.models`.
    """

    system: str | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    reasoning_effort: str | None = None
    output_schema: Mapping[str, JsonValue] | None = None
    extensions: Mapping[str, JsonValue] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def of(
        cls,
        *,
        system: str | None,
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        temperature: float | None,
        top_p: float | None,
        max_output_tokens: int | None,
        stop: Sequence[str],
        seed: int | None,
        reasoning_effort: str | None,
        output_schema: Mapping[str, JsonValue] | None,
        extensions: Mapping[str, JsonValue] | None,
    ) -> "CallOptions":
        """Builds options from public-call arguments, copying every sequence.

        Raises:
            TypeError: ``stop`` is a single string instead of a sequence.
        """
        if isinstance(stop, str):
            raise TypeError("stop must be a sequence of strings, not a string")
        return cls(
            system=system,
            tools=tuple(tools),
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            stop=tuple(stop),
            seed=seed,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            extensions=MappingProxyType(dict(extensions or {})),
        )

    def request(self, prompt: Prompt) -> Request:
        """Returns the request asking ``prompt`` with these options.

        Raises:
            TypeError: ``prompt`` is neither text nor a sequence of `Message`.
            ValueError: The options or the conversation are invalid.
        """
        sampling = Sampling(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            stop=self.stop,
            seed=self.seed,
            reasoning_effort=self.reasoning_effort,
        )
        return Request(
            messages=_messages(prompt),
            system=self.system,
            tools=self.tools,
            tool_choice=self.tool_choice,
            sampling=sampling,
            output_schema=self.output_schema,
            extensions=self.extensions,
        )


def _messages(prompt: Prompt) -> tuple[Message, ...]:
    if isinstance(prompt, str):
        return (Message.user(prompt),)
    messages = tuple(prompt)
    for message in messages:
        if not isinstance(message, Message):
            raise TypeError(f"prompt must be text or Messages, got {type(message).__name__}")
    return messages


def check_settings(spec: ModelSpec, sampling: Sampling) -> None:
    """Rejects settings the model does not accept, before anything is sent.

    A model the catalog does not list has no known settings, so nothing is
    checked; the provider rejects what it does not accept.

    Raises:
        ValueError: A setting the model does not accept, a reasoning effort it
            does not offer, or more output tokens than it can produce.
    """
    if not spec.known:
        return
    rejected = sorted(_settings_used(sampling) - spec.accepts)
    if rejected:
        accepted = ", ".join(sorted(spec.accepts)) or "none"
        raise ValueError(f"{spec.ref} does not accept {', '.join(rejected)}; it accepts {accepted}")
    effort = sampling.reasoning_effort
    if effort is not None and effort not in spec.reasoning_efforts:
        allowed = ", ".join(sorted(spec.reasoning_efforts))
        raise ValueError(f"{spec.ref} has no reasoning_effort {effort!r}; it offers {allowed}")
    limit = spec.max_output_tokens
    requested = sampling.max_output_tokens
    if requested is not None and limit is not None and requested > limit:
        raise ValueError(f"{spec.ref} produces at most {limit} output tokens, not {requested}")


def _settings_used(sampling: Sampling) -> frozenset[SamplingField]:
    used: set[SamplingField] = set()
    if sampling.temperature is not None:
        used.add("temperature")
    if sampling.top_p is not None:
        used.add("top_p")
    if sampling.max_output_tokens is not None:
        used.add("max_output_tokens")
    if sampling.stop:
        used.add("stop")
    if sampling.seed is not None:
        used.add("seed")
    if sampling.reasoning_effort is not None:
        used.add("reasoning_effort")
    return frozenset(used)


class Client:
    """Sends model calls through one catalog, provider pool and limiter.

    A client owns its providers' HTTP connections; close it when done, or use
    it as a context manager. `ember.models` uses a default client, so most
    programs never build one.

    Example:
        >>> from ember.llm import Client
        >>> from ember.llm.fake import FakeProvider, Reply
        >>> fake = FakeProvider(default=Reply("hello"))
        >>> with Client(provider_factory=fake.factory) as client:
        ...     print(client.complete("openai/gpt-5.5", "Say hello.").text)
        hello
    """

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory = providers.build,
        catalog: Catalog | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        """Creates a client. It reads nothing and connects nowhere until first used.

        Args:
            provider_factory: Builds a provider from its settings.
            catalog: Model catalog; ``None`` uses the packaged one.
            sleep: Waits between retries.
            jitter: Returns a uniform random number in [0, 1) for backoff.
        """
        self._factory = provider_factory
        self._catalog = catalog
        self._sleep = sleep
        self._jitter = jitter
        self._pool = ProviderPool(provider_factory)
        self._limiter = Limiter()
        _track(self)

    def complete(
        self,
        model: ModelName,
        prompt: Prompt,
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] = (),
        seed: int | None = None,
        reasoning_effort: str | None = None,
        output_schema: Mapping[str, JsonValue] | None = None,
        extensions: Mapping[str, JsonValue] | None = None,
    ) -> Completion:
        """Calls ``model`` with ``prompt``; the arguments are those of `ember.models`."""
        options = CallOptions.of(
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            stop=stop,
            seed=seed,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            extensions=extensions,
        )
        return self.send(model, options.request(prompt))

    def stream(
        self,
        model: ModelName,
        prompt: Prompt,
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] = (),
        seed: int | None = None,
        reasoning_effort: str | None = None,
        output_schema: Mapping[str, JsonValue] | None = None,
        extensions: Mapping[str, JsonValue] | None = None,
    ) -> Iterator[Event]:
        """Streams ``model``'s answer to ``prompt`` as events, ending with `Done`."""
        options = CallOptions.of(
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            stop=stop,
            seed=seed,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            extensions=extensions,
        )
        return self.send_stream(model, options.request(prompt))

    def bind(
        self,
        model: ModelName,
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] = (),
        seed: int | None = None,
        reasoning_effort: str | None = None,
        output_schema: Mapping[str, JsonValue] | None = None,
        extensions: Mapping[str, JsonValue] | None = None,
    ) -> "BoundModel":
        """Returns a callable that sends prompts to ``model`` through this client."""
        options = CallOptions.of(
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            stop=stop,
            seed=seed,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            extensions=extensions,
        )
        return BoundModel(model=model, options=options, client=self)

    def send(self, model: ModelName, request: Request) -> Completion:
        """Sends a prepared request; see the module docstring for the steps."""
        call = self._prepare(model, request)
        try:
            future = runtime.dispatch(call.run, label=str(call.spec.ref))
        except BaseException:
            call.finish()  # The task never started; return its permit and lease.
            raise
        completion = Completion(future, call.spec.ref)
        if not runtime.deferred():
            runtime.wait(future)
        return completion

    def send_stream(self, model: ModelName, request: Request) -> Iterator[Event]:
        """Streams a prepared request; checks raise here, before anything is sent."""
        call = self._prepare(model, request)
        try:
            return runtime.dispatch_stream(call.open_events, label=str(call.spec.ref))
        except BaseException:
            call.finish()  # The stream never started; return its permit and lease.
            raise

    def limiter_state(self, provider: str) -> LimiterState | None:
        """Returns ``provider``'s current limit and load, or ``None`` before its first call."""
        return self._limiter.state(provider)

    def close(self) -> None:
        """Closes every provider this client built. Later calls raise RuntimeError."""
        self._pool.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def check(self, model: ModelName, *, sampling: Sampling = DEFAULT_SAMPLING) -> ModelSpec:
        """Raises what a call to ``model`` with ``sampling`` would raise before sending.

        Resolves the model, checks its settings and the price policy, and
        obtains the provider from this client's pool with this client's
        factory, exactly as a call does, so a missing key or base URL raises
        here too. It builds, and keeps pooled, the provider the next call
        would build; it opens no connection, takes no capacity permit and
        sends nothing.

        Returns:
            The resolved spec.

        Raises:
            ModelError: ``NOT_FOUND`` for an unknown model, ``UNPRICED`` under
                ``require_price``.
            ValueError: A setting the model does not accept.
            ConfigError: No model given and no ``default_model``, the provider
                is not configured, or it cannot be built (a missing key or
                base URL).
        """
        ctx = context.current()
        spec, settings, _ = self._preflight(model, sampling, keeps_list_price=True, ctx=ctx)
        self._pool.lease(settings, ctx).release()
        return spec

    def _prepare(self, model: ModelName, request: Request) -> Call:
        ctx = context.current()
        spec, settings, list_price = self._preflight(
            model,
            request.sampling,
            keeps_list_price=extensions_keep_list_price(request),
            ctx=ctx,
        )
        lease = self._pool.lease(settings, ctx)
        try:
            permit = self._wait_for_permit(spec.ref.provider, ctx.max_in_flight_per_provider)
        except BaseException:
            lease.release()
            raise
        return Call(
            call_id=uuid.uuid4().hex,
            spec=spec,
            request=request,
            lease=lease,
            permit=permit,
            retry=ctx.retry,
            reporter=observe.reporter(),
            list_price=list_price,
            sleep=self._sleep,
            jitter=self._jitter,
        )

    def _preflight(
        self,
        model: ModelName,
        sampling: Sampling,
        *,
        keeps_list_price: bool,
        ctx: Context,
    ) -> tuple[ModelSpec, ProviderSettings, bool]:
        """Every check a call makes before sending: model, settings, provider, price."""
        spec = self._resolve(model, ctx)
        check_settings(spec, sampling)
        settings = ctx.provider(spec.ref.provider)
        list_price = settings.base_url is None and keeps_list_price
        _check_price(spec, list_price, ctx)
        return spec, settings, list_price

    def _resolve(self, model: ModelName, ctx: Context) -> ModelSpec:
        if model is None:
            if ctx.default_model is None:
                raise ConfigError("no model given and no default_model configured")
            model = ctx.default_model
        table = self._catalog if self._catalog is not None else catalog_module.default()
        return table.resolve(model, providers=ctx.providers)

    def _wait_for_permit(self, provider: str, cap: int) -> Permit:
        pending = self._limiter.acquire(provider, cap=cap)
        try:
            return runtime.wait(pending)
        except BaseException:
            # Stop waiting; a permit granted in the meantime goes straight back.
            if not pending.cancel():
                pending.result().release()
            raise

    def _reset_after_fork(self) -> None:
        # The parent's connections and locks are unusable in the child; drop
        # them without closing, which would disturb the parent's sockets.
        self._pool = ProviderPool(self._factory)
        self._limiter = Limiter()


def _check_price(spec: ModelSpec, list_price: bool, ctx: Context) -> None:
    if not ctx.require_price:
        return
    if spec.rates is None:
        reason = "the model has no published price and require_price is set"
    elif not list_price:
        reason = "a custom base_url or an extension may change the price and require_price is set"
    else:
        return
    raise ModelError(
        ErrorKind.UNPRICED, provider=spec.ref.provider, model=spec.ref.name, reason=reason
    )


@dataclasses.dataclass(frozen=True, slots=True)
class BoundModel:
    """A model with fixed call options, called with just a prompt.

    Attributes:
        model: The model, resolved again at every call so the call's context
            decides the provider settings.
        options: The options every call uses.
        client: The client calls go to; ``None`` means the client active at
            call time (see `using`), which is what `ember.models.bind` gives.
    """

    model: ModelName
    options: CallOptions
    client: Client | None = None

    def __call__(self, prompt: Prompt) -> Completion:
        """Sends ``prompt``; behaves like `ember.models` with the bound options."""
        return self._client().send(self.model, self.options.request(prompt))

    def stream(self, prompt: Prompt) -> Iterator[Event]:
        """Streams the answer to ``prompt``."""
        return self._client().send_stream(self.model, self.options.request(prompt))

    def _client(self) -> Client:
        return self.client if self.client is not None else active()


# Which client `ember.models` uses. Owner: this module.
_USING: contextvars.ContextVar[Client] = contextvars.ContextVar("ember_client")

# The default client: built on first use under `_default_lock`, closed at
# interpreter exit, discarded in a forked child.
_default_lock = threading.Lock()
_default: Client | None = None

# Every live client, so a forked child can discard their connections.
_live_lock = threading.Lock()
_live: "weakref.WeakSet[Client]" = weakref.WeakSet()


def active() -> Client:
    """Returns the client `ember.models` sends to: the `using` one, else the default."""
    chosen = _USING.get(None)
    return chosen if chosen is not None else _default_client()


@contextlib.contextmanager
def using(client: Client) -> Generator[Client, None, None]:
    """Sends every `ember.models` call made in the block through ``client``.

    Example:
        >>> import ember
        >>> from ember.llm import Client, using
        >>> from ember.llm.fake import FakeProvider, Reply
        >>> fake = FakeProvider(default=Reply("4"))
        >>> with Client(provider_factory=fake.factory) as client, using(client):
        ...     print(ember.models("openai/gpt-5.5", "What is 2 + 2?").text)
        4
    """
    token = _USING.set(client)
    try:
        yield client
    finally:
        _USING.reset(token)


def _default_client() -> Client:
    global _default
    client = _default
    if client is not None:
        return client
    with _default_lock:
        if _default is None:
            _default = Client()
        return _default


def _close_default() -> None:
    global _default
    with _default_lock:
        client, _default = _default, None
    if client is not None:
        client.close()


def _track(client: Client) -> None:
    with _live_lock:
        _live.add(client)


def _after_fork_in_child() -> None:
    global _default, _default_lock, _live_lock
    _default_lock = threading.Lock()
    _live_lock = threading.Lock()
    _default = None
    for client in list(_live):
        client._reset_after_fork()


atexit.register(_close_default)
os.register_at_fork(after_in_child=_after_fork_in_child)
