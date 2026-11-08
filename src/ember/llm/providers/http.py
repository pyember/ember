"""The HTTP transport every provider uses.

One `Transport` owns one long-lived ``httpx.Client`` for one provider
configuration. Every request streams, so its timeouts are liveness bounds:
how long to wait for a connection and how long to wait for the next byte.
Nothing caps how long a generation may take. A model may stay silent for many
minutes while it reasons, so the idle bound is long; TCP keepalive on every
connection is what notices a peer that died in the meantime. The transport never retries; it
turns every HTTP or network failure into a `ModelError` whose message is
built from the status, the provider's short error code and an ember-written
reason, never from the upstream body.
"""

import contextlib
import dataclasses
import email.utils
import json
import math
import socket
import sys
import time
from collections.abc import Callable, Generator, Iterator, Mapping

import httpx

from ember.errors import ErrorKind, ModelError
from ember.llm.providers.sse import SseEvent, parse
from ember.llm.providers.wire import Body, Failure

__all__ = ["Endpoint", "EventStream", "Transport", "keepalive_socket_options"]

_REASONS: Mapping[ErrorKind, str] = {
    ErrorKind.AUTH: "the provider rejected the credentials",
    ErrorKind.PERMISSION: "the credentials are not allowed to make this request",
    ErrorKind.QUOTA: "the account's credit, spend limit or usage quota is exhausted",
    ErrorKind.NOT_FOUND: "the provider does not know this model or endpoint",
    ErrorKind.BAD_REQUEST: "the provider rejected the request as invalid",
    ErrorKind.RATE_LIMITED: "the provider rate-limited the request",
    ErrorKind.OVERLOADED: "the provider is overloaded",
    ErrorKind.SERVER: "the provider failed to serve the request",
    ErrorKind.MALFORMED: "the provider's error response could not be read",
    ErrorKind.TRANSPORT: "the provider rejected the connection",
    ErrorKind.CANCELLED: "the provider cancelled the request",
}

# The httpx failures that mean the network or the peer failed. Anything else
# httpx raises (an invalid URL, a protocol violation on our side) is a bug in
# ember or its configuration and propagates unchanged.
_NETWORK_FAILURES = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
    httpx.DecodingError,
)

# TCP keepalive: first probe after 60 s of silence, then every 15 s, and the
# connection is declared dead after 4 unanswered probes (about two minutes).
_KEEPALIVE_IDLE_S = 60
_KEEPALIVE_INTERVAL_S = 15
_KEEPALIVE_PROBES = 4

Classifier = Callable[[int, object], Failure]
"""Maps an HTTP error's status and JSON body (or ``None``) to a `Failure`."""


@dataclasses.dataclass(frozen=True, slots=True)
class Endpoint:
    """Where and how to reach one provider.

    Attributes:
        provider: Provider name used in errors.
        base_url: Root that request paths are joined to.
        headers: Headers sent with every request, authentication included.
            They are given to the HTTP client and never appear in errors.
        request_id_headers: Response headers that carry the provider's
            request id, in the order they are consulted.
    """

    provider: str
    base_url: str
    headers: Mapping[str, str]
    request_id_headers: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class EventStream:
    """A successful streaming response that is still being read."""

    events: Iterator[SseEvent]
    request_id: str | None


class Transport:
    """Streams JSON requests to one provider over one owned HTTP client."""

    def __init__(
        self,
        endpoint: Endpoint,
        *,
        connect_timeout_s: float,
        idle_timeout_s: float,
        max_connections: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Creates the HTTP client.

        Args:
            endpoint: The provider to talk to.
            connect_timeout_s: Bound on establishing a connection, and on
                waiting for a pooled one.
            idle_timeout_s: Bound on waiting for the next byte in either
                direction. Not a cap on total generation time.
            max_connections: Connections kept per provider; the client's
                limiter keeps requests in flight at or below this.
            transport: Replaces the network, for tests. Without it, connections
                use TCP keepalive (`keepalive_socket_options`).
        """
        self._endpoint = endpoint
        limits = httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        )
        if transport is None:
            transport = httpx.HTTPTransport(
                limits=limits, socket_options=keepalive_socket_options()
            )
        self._client = httpx.Client(
            base_url=endpoint.base_url,
            headers=dict(endpoint.headers),
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                read=idle_timeout_s,
                write=idle_timeout_s,
                pool=connect_timeout_s,
            ),
            limits=limits,
            transport=transport,
        )

    @contextlib.contextmanager
    def stream(
        self,
        path: str,
        body: Body,
        *,
        model: str,
        classify: Classifier,
        params: Mapping[str, str] | None = None,
    ) -> Generator[EventStream, None, None]:
        """Sends ``body`` and yields the response's events as they arrive.

        Leaving the block closes the response and frees its connection, so a
        consumer that stops reading early does not leak it.

        Args:
            path: Path relative to the base URL.
            body: The JSON request body.
            model: Model name for errors.
            classify: The wire protocol's reading of an HTTP error body.
            params: Query parameters.

        Raises:
            ModelError: The request failed, or the response is not an event
                stream.
        """
        request = self._build(path, body, params)
        try:
            response = self._client.send(request, stream=True)
        except _NETWORK_FAILURES as error:
            raise self._network_error(error, model, None, before_first_byte=True) from None
        try:
            request_id = self._request_id(response)
            if not response.is_success:
                raise self._status_error(response, model, request_id, classify)
            if not response.headers.get("content-type", "").startswith("text/event-stream"):
                raise self._error(
                    Failure(ErrorKind.MALFORMED),
                    "the provider did not return an event stream; ember streams every "
                    "call, so the server must support streaming responses",
                    model,
                    response,
                    request_id,
                )
            yield EventStream(
                events=self._events(response, model, request_id), request_id=request_id
            )
        finally:
            response.close()

    def close(self) -> None:
        """Closes the HTTP client and its connections."""
        self._client.close()

    def _build(self, path: str, body: Body, params: Mapping[str, str] | None) -> httpx.Request:
        content = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return self._client.build_request(
            "POST",
            path,
            params=dict(params) if params else None,
            content=content.encode("utf-8"),
            headers={"content-type": "application/json", "accept": "text/event-stream"},
        )

    def _events(
        self, response: httpx.Response, model: str, request_id: str | None
    ) -> Iterator[SseEvent]:
        try:
            yield from parse(response.iter_bytes())
        except _NETWORK_FAILURES as error:
            raise self._network_error(error, model, request_id, before_first_byte=False) from None
        except UnicodeDecodeError:
            raise ModelError(
                ErrorKind.MALFORMED,
                provider=self._endpoint.provider,
                model=model,
                reason="the event stream is not valid UTF-8",
                request_id=request_id,
                before_first_byte=False,
            ) from None

    def _request_id(self, response: httpx.Response) -> str | None:
        for header in self._endpoint.request_id_headers:
            value: str | None = response.headers.get(header)
            if value:
                return value
        return None

    def _status_error(
        self,
        response: httpx.Response,
        model: str,
        request_id: str | None,
        classify: Classifier,
    ) -> ModelError:
        try:
            response.read()
        except _NETWORK_FAILURES as error:
            return self._network_error(error, model, request_id, before_first_byte=True)
        failure = classify(response.status_code, _json_or_none(response.content))
        return self._error(failure, _REASONS[failure.kind], model, response, request_id)

    def _error(
        self,
        failure: Failure,
        reason: str,
        model: str,
        response: httpx.Response,
        request_id: str | None,
    ) -> ModelError:
        retry_after_s = _retry_after_s(response.headers.get("retry-after"))
        return ModelError(
            failure.kind,
            provider=self._endpoint.provider,
            model=model,
            reason=failure.describe(reason),
            status=response.status_code,
            request_id=request_id,
            retry_after_s=failure.retry_after_s if retry_after_s is None else retry_after_s,
        )

    def _network_error(
        self, error: Exception, model: str, request_id: str | None, *, before_first_byte: bool
    ) -> ModelError:
        when = "before the response started" if before_first_byte else "while reading the response"
        return ModelError(
            ErrorKind.TRANSPORT,
            provider=self._endpoint.provider,
            model=model,
            reason=f"{type(error).__name__} {when}",
            request_id=request_id,
            before_first_byte=before_first_byte,
        )


def keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Returns the socket options that turn on TCP keepalive.

    Linux and macOS also get the probe timing; other platforms keep their
    system timing, which only ``SO_KEEPALIVE`` is portable enough to set.
    """
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if sys.platform == "linux":
        idle = socket.TCP_KEEPIDLE
    elif sys.platform == "darwin":
        idle = socket.TCP_KEEPALIVE
    else:
        return options
    return [
        *options,
        (socket.IPPROTO_TCP, idle, _KEEPALIVE_IDLE_S),
        (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, _KEEPALIVE_INTERVAL_S),
        (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, _KEEPALIVE_PROBES),
    ]


def _json_or_none(content: bytes) -> object:
    try:
        value: object = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value


def _retry_after_s(value: str | None) -> float | None:
    """Parses ``Retry-After`` as delay-seconds or an HTTP date (RFC 9110 10.2.3).

    The header is advice, so a value that is neither is ignored.
    """
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            moment = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = moment.timestamp() - time.time()
    if not math.isfinite(seconds):
        return None
    return max(seconds, 0.0)
