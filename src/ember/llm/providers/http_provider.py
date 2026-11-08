"""A provider: a transport plus the codec each model's wire protocol needs.

`HttpProvider` implements `ember.llm.provider.Provider` for every shipped
provider kind; the kinds differ only in their `Endpoint` and in how a model
is routed to a path and codec. It encodes before sending, so an invalid
request raises before any byte leaves, and it is the one place a codec's
`WireError` becomes a `ModelError` with the provider, model and request id.
"""

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType

from ember.errors import ModelError
from ember.llm.provider import Decoded, StreamItem
from ember.llm.providers.http import Transport
from ember.llm.providers.wire import Body, Codec, WireError
from ember.llm.request import Request
from ember.llm.spec import ModelSpec

__all__ = ["HttpProvider", "Route", "Router"]


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


Router = Callable[[ModelSpec, bool], Route]
"""Chooses the route for ``(spec, stream)``; raises `ConfigError` for a model
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
        """Sends ``request`` and returns the decoded response.

        Raises:
            ValueError: The request cannot be expressed in the model's wire
                protocol, or ``spec`` belongs to another provider.
            ModelError: The call failed.
        """
        route, body = self._prepare(spec, request, stream=False)
        response = self._transport.post(route.path, body, model=spec.ref.name, params=route.params)
        try:
            return route.codec.decode(spec, response.body, response.request_id)
        except WireError as error:
            raise error.to_model_error(
                provider=self._name,
                model=spec.ref.name,
                request_id=response.request_id,
                before_first_byte=True,
            ) from error

    def stream(self, spec: ModelSpec, request: Request) -> Iterator[StreamItem]:
        """Sends ``request`` with streaming; the request is validated before
        this returns, and sent when the first item is read.

        Raises:
            ValueError: As in `send`, raised by this call itself.
            ModelError: Raised while iterating, when the call fails.
        """
        route, body = self._prepare(spec, request, stream=True)
        return self._items(spec, route, body)

    def close(self) -> None:
        """Releases the provider's HTTP connections."""
        self._transport.close()

    def _prepare(self, spec: ModelSpec, request: Request, *, stream: bool) -> tuple[Route, Body]:
        if spec.ref.provider != self._name:
            raise ValueError(f"model {spec.ref} is not served by provider {self._name!r}")
        route = self._router(spec, stream)
        return route, route.codec.encode(spec, request, stream)

    def _items(self, spec: ModelSpec, route: Route, body: Body) -> Iterator[StreamItem]:
        """Yields the stream's items; a failure says whether any reached the caller.

        `ModelError.before_first_byte` means the caller received nothing from
        this attempt, so it stays True for a failure that arrives before the
        first item even when response bytes had already been read.
        """
        delivered = False
        try:
            with self._transport.stream(
                route.path, body, model=spec.ref.name, params=route.params
            ) as opened:
                try:
                    for item in route.codec.decode_events(spec, opened.events, opened.request_id):
                        delivered = True
                        yield item
                except WireError as error:
                    raise error.to_model_error(
                        provider=self._name,
                        model=spec.ref.name,
                        request_id=opened.request_id,
                        before_first_byte=not delivered,
                    ) from error
        except ModelError as error:
            if delivered or error.before_first_byte:
                raise
            raise _undelivered(error) from error


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
