"""The transport classifies every failure and never leaks upstream text."""

import email.utils
import time
from collections.abc import Callable, Iterator

import httpx
import pytest

from ember.errors import ErrorKind, ModelError
from ember.llm.providers import openai_chat
from ember.llm.providers.http import Endpoint, Transport

Handler = Callable[[httpx.Request], httpx.Response]


def _transport(handler: Handler) -> Transport:
    endpoint = Endpoint(
        provider="acme",
        base_url="https://api.example.test/v1",
        headers={"authorization": "Bearer test-key"},
        request_id_headers=("x-request-id", "apim-request-id"),
        error_label=openai_chat.error_label,
    )
    return Transport(
        endpoint,
        connect_timeout_s=3.0,
        idle_timeout_s=7.0,
        max_connections=4,
        transport=httpx.MockTransport(handler),
    )


def _failure(response: httpx.Response) -> ModelError:
    with pytest.raises(ModelError) as caught:
        _transport(lambda request: response).post("chat/completions", {}, model="m")
    return caught.value


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (400, ErrorKind.BAD_REQUEST, False),
        (401, ErrorKind.AUTH, False),
        (402, ErrorKind.PERMISSION, False),
        (403, ErrorKind.PERMISSION, False),
        (404, ErrorKind.NOT_FOUND, False),
        (408, ErrorKind.SERVER, True),
        (409, ErrorKind.BAD_REQUEST, False),
        (413, ErrorKind.BAD_REQUEST, False),
        (422, ErrorKind.BAD_REQUEST, False),
        (429, ErrorKind.RATE_LIMITED, True),
        (500, ErrorKind.SERVER, True),
        (502, ErrorKind.SERVER, True),
        (503, ErrorKind.OVERLOADED, True),
        (504, ErrorKind.SERVER, True),
        (529, ErrorKind.OVERLOADED, True),
    ],
)
def test_statuses_map_to_kinds(status: int, kind: ErrorKind, retryable: bool) -> None:
    error = _failure(httpx.Response(status, json={}))
    assert (error.kind, error.status, error.retryable) == (kind, status, retryable)
    assert error.provider == "acme" and error.model == "m"


def test_errors_carry_the_provider_code_and_request_id_but_never_the_body() -> None:
    body = {
        "error": {
            "message": "upstream overloaded on shard us-east-7 for key sk-live-abc",
            "type": "server_error",
            "code": "server_error",
        }
    }
    error = _failure(httpx.Response(500, json=body, headers={"x-request-id": "req_123"}))
    assert error.request_id == "req_123"
    assert "server_error" in str(error)
    assert "shard" not in str(error) and "sk-live" not in str(error)


def test_free_text_in_code_fields_is_not_copied() -> None:
    body = {"error": {"code": "failed at 10.0.0.7 because the shard is hot", "type": None}}
    error = _failure(httpx.Response(400, json=body))
    assert "10.0.0.7" not in str(error)


def test_azure_request_id_header_is_read_when_it_is_the_only_one() -> None:
    error = _failure(httpx.Response(401, json={}, headers={"apim-request-id": "az-1"}))
    assert error.request_id == "az-1"


@pytest.mark.parametrize(
    ("header", "expected"),
    [("2", 2.0), ("1.5", 1.5), ("-3", 0.0), ("nan", None), ("inf", None), ("soon", None)],
)
def test_retry_after_seconds(header: str, expected: float | None) -> None:
    error = _failure(httpx.Response(429, json={}, headers={"retry-after": header}))
    assert error.retry_after_s == expected


def test_retry_after_http_date() -> None:
    later = email.utils.formatdate(time.time() + 30, usegmt=True)
    error = _failure(httpx.Response(503, json={}, headers={"retry-after": later}))
    assert error.retry_after_s is not None and 25 <= error.retry_after_s <= 30


def test_timeouts_are_connect_and_idle_bounds_only() -> None:
    seen: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={})

    _transport(handler).post("chat/completions", {}, model="m")
    # read and write are per-byte (idle) bounds; httpx has no total deadline.
    assert seen == [{"connect": 3.0, "read": 7.0, "write": 7.0, "pool": 3.0}]


def test_connection_failure_is_a_retryable_transport_error_without_os_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused to 10.1.2.3")

    with pytest.raises(ModelError) as caught:
        _transport(handler).post("chat/completions", {}, model="m")
    error = caught.value
    assert (error.kind, error.before_first_byte, error.retryable) == (
        ErrorKind.TRANSPORT,
        True,
        True,
    )
    assert "10.1.2.3" not in str(error)


def test_non_json_success_body_is_a_server_error() -> None:
    error = _failure(httpx.Response(200, content=b"<html>gateway</html>"))
    assert error.kind is ErrorKind.SERVER and "gateway" not in str(error)


def test_unserializable_body_raises_before_sending() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    with pytest.raises(ValueError):
        _transport(handler).post("x", {"temperature": float("nan")}, model="m")
    assert sent == []


class _Body(httpx.SyncByteStream):
    """A response body that can fail after some bytes and records closing."""

    def __init__(self, chunks: list[bytes], fail: bool = False) -> None:
        self.chunks, self.fail, self.closed = chunks, fail, False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks
        if self.fail:
            raise httpx.ReadError("connection reset by peer 10.9.9.9")

    def close(self) -> None:
        self.closed = True


def _stream_response(body: _Body, content_type: str = "text/event-stream") -> Handler:
    return lambda request: httpx.Response(200, headers={"content-type": content_type}, stream=body)


def test_read_failure_mid_stream_is_transport_after_the_first_byte() -> None:
    body = _Body([b'data: {"a":1}\n\n'], fail=True)
    seen = []
    with pytest.raises(ModelError) as caught:
        with _transport(_stream_response(body)).stream("x", {}, model="m") as opened:
            for event in opened.events:
                seen.append(event.data)
    assert seen == ['{"a":1}']
    error = caught.value
    assert (error.kind, error.before_first_byte, error.retryable) == (
        ErrorKind.TRANSPORT,
        False,
        False,
    )


def test_abandoned_stream_closes_the_response() -> None:
    body = _Body([b"data: 1\n\n", b"data: 2\n\n"])
    with _transport(_stream_response(body)).stream("x", {}, model="m") as opened:
        next(opened.events)
    assert body.closed


def test_stream_must_be_an_event_stream() -> None:
    body = _Body([b'{"choices": []}'])
    with pytest.raises(ModelError) as caught:
        with _transport(_stream_response(body, "application/json")).stream("x", {}, model="m"):
            pass
    assert caught.value.kind is ErrorKind.SERVER
    assert body.closed


def test_stream_error_status_is_classified_with_its_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"error": {"type": "rate_limit_error", "message": "slow down, org 42"}}
        return httpx.Response(429, json=body, headers={"retry-after": "4"})

    with pytest.raises(ModelError) as caught:
        with _transport(handler).stream("x", {}, model="m"):
            pass
    error = caught.value
    assert (error.kind, error.retry_after_s) == (ErrorKind.RATE_LIMITED, 4.0)
    assert "rate_limit_error" in str(error) and "org 42" not in str(error)
