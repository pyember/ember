"""The HTTP transport every provider uses.

One `Transport` owns one long-lived ``httpx.Client`` for one provider
configuration. Its timeouts are liveness bounds only: how long to wait for a
connection and how long to wait for the next byte. Nothing caps how long a
generation may take. The transport never retries; it turns every HTTP or
network failure into a `ModelError` whose message is built from the status,
the provider's short error code and an ember-written reason, never from the
upstream body.
"""

import contextlib
import dataclasses
import email.utils
import json
import math
import time
from collections.abc import Callable, Generator, Iterator, Mapping

import httpx

from ember.errors import ErrorKind, ModelError
from ember.llm.providers.sse import SseEvent, parse
from ember.llm.providers.wire import Body

__all__ = ["Endpoint", "EventStream", "JsonResponse", "Transport", "kind_for_status"]

_STATUS_KINDS: Mapping[int, ErrorKind] = {
    400: ErrorKind.BAD_REQUEST,
    401: ErrorKind.AUTH,
    402: ErrorKind.PERMISSION,
    403: ErrorKind.PERMISSION,
    404: ErrorKind.NOT_FOUND,
    408: ErrorKind.SERVER,
    413: ErrorKind.BAD_REQUEST,
    422: ErrorKind.BAD_REQUEST,
    429: ErrorKind.RATE_LIMITED,
    503: ErrorKind.OVERLOADED,
    529: ErrorKind.OVERLOADED,
}

_REASONS: Mapping[ErrorKind, str] = {
    ErrorKind.AUTH: "the provider rejected the credentials",
    ErrorKind.PERMISSION: "the credentials are not allowed to make this request",
    ErrorKind.NOT_FOUND: "the provider does not know this model or endpoint",
    ErrorKind.BAD_REQUEST: "the provider rejected the request as invalid",
    ErrorKind.RATE_LIMITED: "the provider rate-limited the request",
    ErrorKind.OVERLOADED: "the provider is overloaded",
    ErrorKind.SERVER: "the provider failed to serve the request",
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


def kind_for_status(status: int) -> ErrorKind:
    """Maps a non-success HTTP status to the kind of failure it reports."""
    if status in _STATUS_KINDS:
        return _STATUS_KINDS[status]
    if 400 <= status < 500:
        return ErrorKind.BAD_REQUEST
    return ErrorKind.SERVER


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
        error_label: Extracts the provider's short error type or code from
            an error response body.
    """

    provider: str
    base_url: str
    headers: Mapping[str, str]
    request_id_headers: tuple[str, ...]
    error_label: Callable[[object], str | None]


@dataclasses.dataclass(frozen=True, slots=True)
class JsonResponse:
    """A successful non-streaming response."""

    body: object
    request_id: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class EventStream:
    """A successful streaming response that is still being read."""

    events: Iterator[SseEvent]
    request_id: str | None


class Transport:
    """Sends JSON requests to one provider over one owned HTTP client."""

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
            transport: Replaces the network, for tests.
        """
        self._endpoint = endpoint
        self._client = httpx.Client(
            base_url=endpoint.base_url,
            headers=dict(endpoint.headers),
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                read=idle_timeout_s,
                write=idle_timeout_s,
                pool=connect_timeout_s,
            ),
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
            transport=transport,
        )

    def post(
        self, path: str, body: Body, *, model: str, params: Mapping[str, str] | None = None
    ) -> JsonResponse:
        """Sends ``body`` and returns the decoded JSON response.

        Raises:
            ModelError: The request failed, or the response is not JSON.
        """
        request = self._build(path, body, params, accept="application/json")
        try:
            response = self._client.send(request)
        except _NETWORK_FAILURES as error:
            raise self._network_error(error, model, before_first_byte=True) from None
        request_id = self._request_id(response)
        if not response.is_success:
            raise self._status_error(response, model, request_id)
        try:
            body_value: object = json.loads(response.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise self._error(
                ErrorKind.SERVER, model, "the response body is not JSON", response, request_id
            ) from None
        return JsonResponse(body=body_value, request_id=request_id)

    @contextlib.contextmanager
    def stream(
        self, path: str, body: Body, *, model: str, params: Mapping[str, str] | None = None
    ) -> Generator[EventStream, None, None]:
        """Sends ``body`` and yields the response's events as they arrive.

        Leaving the block closes the response and frees its connection, so a
        consumer that stops reading early does not leak it.

        Raises:
            ModelError: The request failed before or during the stream.
        """
        request = self._build(path, body, params, accept="text/event-stream")
        try:
            response = self._client.send(request, stream=True)
        except _NETWORK_FAILURES as error:
            raise self._network_error(error, model, before_first_byte=True) from None
        try:
            request_id = self._request_id(response)
            if not response.is_success:
                self._read_quietly(response, model)
                raise self._status_error(response, model, request_id)
            if not response.headers.get("content-type", "").startswith("text/event-stream"):
                raise self._error(
                    ErrorKind.SERVER,
                    model,
                    "the provider did not return an event stream",
                    response,
                    request_id,
                )
            yield EventStream(events=self._events(response, model), request_id=request_id)
        finally:
            response.close()

    def close(self) -> None:
        """Closes the HTTP client and its connections."""
        self._client.close()

    def _build(
        self, path: str, body: Body, params: Mapping[str, str] | None, *, accept: str
    ) -> httpx.Request:
        content = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return self._client.build_request(
            "POST",
            path,
            params=dict(params) if params else None,
            content=content.encode("utf-8"),
            headers={"content-type": "application/json", "accept": accept},
        )

    def _events(self, response: httpx.Response, model: str) -> Iterator[SseEvent]:
        try:
            yield from parse(response.iter_bytes())
        except _NETWORK_FAILURES as error:
            raise self._network_error(error, model, before_first_byte=False) from None
        except UnicodeDecodeError:
            raise ModelError(
                ErrorKind.SERVER,
                provider=self._endpoint.provider,
                model=model,
                reason="the event stream is not valid UTF-8",
                request_id=self._request_id(response),
                before_first_byte=False,
            ) from None

    def _read_quietly(self, response: httpx.Response, model: str) -> None:
        """Reads an error response's body so its error code can be labelled."""
        try:
            response.read()
        except _NETWORK_FAILURES as error:
            raise self._network_error(error, model, before_first_byte=False) from None

    def _request_id(self, response: httpx.Response) -> str | None:
        for header in self._endpoint.request_id_headers:
            value: str | None = response.headers.get(header)
            if value:
                return value
        return None

    def _status_error(
        self, response: httpx.Response, model: str, request_id: str | None
    ) -> ModelError:
        kind = kind_for_status(response.status_code)
        reason = _REASONS[kind]
        label = self._endpoint.error_label(_json_or_none(response.content))
        if label is not None:
            reason = f"{reason}; provider error {label}"
        return self._error(kind, model, reason, response, request_id)

    def _error(
        self,
        kind: ErrorKind,
        model: str,
        reason: str,
        response: httpx.Response,
        request_id: str | None,
    ) -> ModelError:
        return ModelError(
            kind,
            provider=self._endpoint.provider,
            model=model,
            reason=reason,
            status=response.status_code,
            request_id=request_id,
            retry_after_s=_retry_after_s(response.headers.get("retry-after")),
        )

    def _network_error(
        self, error: Exception, model: str, *, before_first_byte: bool
    ) -> ModelError:
        when = "before the response started" if before_first_byte else "while reading the response"
        return ModelError(
            ErrorKind.TRANSPORT,
            provider=self._endpoint.provider,
            model=model,
            reason=f"{type(error).__name__} {when}",
            before_first_byte=before_first_byte,
        )


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
