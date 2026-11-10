"""A provider: a transport plus the codec each model's wire protocol needs.

`HttpProvider` implements `ember.llm.provider.Provider` for every shipped
provider kind; the kinds differ only in their `Endpoint` and in how a model
is routed to a path and codec. Every call streams: `send` reads the stream
to its end and returns the final `Decoded`, so the idle bound stays a
liveness bound however long a generation runs. The provider encodes before
sending, so an invalid request raises before any byte leaves, and it is the
one place a codec's failure becomes a `ModelError` with the provider, model
and request id.
"""

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType

from ember.errors import ErrorKind, ModelError
from ember.llm.provider import Decoded, StreamItem
from ember.llm.providers.http import Transport
from ember.llm.providers.wire import Body, Codec, WireError
from ember.llm.request import Request
from ember.llm.spec import ModelSpec

__all__ = ["HttpProvider", "Route", "Router"]

# What a codec raises when a response breaks an assumption its shape checks
# did not cover. Each is reported as an undecodable response, chained to the
# original so the codec bug stays visible.
_DECODE_FAILURES = (KeyError, IndexError, TypeError, ValueError)


def _no_params() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclasses.dataclass(frozen=True, slots=True)
class Route:
    """Where one request goes and which codec speaks to it.

    Attributes:
        path: Path relative to the provider's base URL.
        codec: The wire protocol at that path.
        params: Query parameters, for example Gemini's ``alt=sse``.
    """

    path: str
    codec: Codec
    params: Mapping[str, str] = dataclasses.field(default_factory=_no_params)


Router = Callable[[ModelSpec], Route]
"""Chooses the streaming route for a model; raises `ConfigError` for a model
whose wire protocol the provider does not speak."""


class HttpProvider:
    """Sends requests for one configured provider."""

    def __init__(self, name: str, transport: Transport, router: Router) -> None:
        """Creates the provider; it owns ``transport`` and closes it in `close`."""
        self._name = name
        self._transport = transport
        self._router = router

    @property
    def name(self) -> str:
        """The provider name used in model ids."""
        return self._name

    def send(self, spec: ModelSpec, request: Request) -> Decoded:
        """Sends ``request``, reads the whole stream and returns the result.

        Nothing reaches the caller before the result, so every failure is
        reported with ``before_first_byte=True``.

        Raises:
            ValueError: The request cannot be expressed in the model's wire
                protocol, or ``spec`` belongs to another provider.
            ModelError: The call failed.
        """
        final: StreamItem | None = None
        try:
            for item in self.stream(spec, request):
                final = item
        except ModelError as error:
            if error.before_first_byte:
                raise
            raise _undelivered(error) from error
        if not isinstance(final, Decoded):
            raise AssertionError("a codec stream ended without its Decoded")
        return final

    def stream(self, spec: ModelSpec, request: Request) -> Iterator[StreamItem]:
        """Sends ``request``; the request is validated before this returns,
        and sent when the first item is read.

        Raises:
            ValueError: As in `send`, raised by this call itself.
            ModelError: Raised while iterating, when the call fails.
        """
        if spec.ref.provider != self._name:
            raise ValueError(f"model {spec.ref} is not served by provider {self._name!r}")
        route = self._router(spec)
        return self._items(spec, request, route, route.codec.encode(spec, request))

    def close(self) -> None:
        """Releases the provider's HTTP connections."""
        self._transport.close()

    def _items(
        self, spec: ModelSpec, request: Request, route: Route, body: Body
    ) -> Iterator[StreamItem]:
        """Yields the stream's items; a failure says whether any reached the caller.

        `ModelError.before_first_byte` means the caller received nothing from
        this attempt, so it stays True for a failure that arrives before the
        first item even when response bytes had already been read. Only the
        codec's own work is classified: an exception the consumer throws in
        at a ``yield`` passes through unchanged.
        """
        delivered = False
        try:
            with self._transport.stream(
                route.path,
                body,
                model=spec.ref.name,
                classify=route.codec.classify,
                params=route.params,
            ) as opened:
                items = route.codec.decode_events(spec, request, opened.events, opened.request_id)
                while True:
                    try:
                        item = next(items)
                    except StopIteration:
                        return
                    except WireError as error:
                        raise self._failed(error, spec, opened.request_id, delivered) from error
                    except _DECODE_FAILURES as error:
                        reason = f"the response could not be decoded ({type(error).__name__})"
                        undecodable = WireError(ErrorKind.MALFORMED, reason)
                        raise self._failed(
                            undecodable, spec, opened.request_id, delivered
                        ) from error
                    delivered = True
                    yield item
        except ModelError as error:
            if delivered or error.before_first_byte:
                raise
            raise _undelivered(error) from error

    def _failed(
        self, error: WireError, spec: ModelSpec, request_id: str | None, delivered: bool
    ) -> ModelError:
        return error.to_model_error(
            provider=self._name,
            model=spec.ref.name,
            request_id=request_id,
            before_first_byte=not delivered,
        )


def _undelivered(error: ModelError) -> ModelError:
    """Returns ``error`` marked as having delivered nothing to the caller."""
    return ModelError(
        error.kind,
        provider=error.provider,
        model=error.model,
        reason=error.reason,
        status=error.status,
        request_id=error.request_id,
        retry_after_s=error.retry_after_s,
        before_first_byte=True,
    )
