"""build(): each provider kind's URL, authentication, routing and failure contract."""

import json
import subprocess
import sys
import textwrap

import httpx
import pytest

from ember.errors import ConfigError, ErrorKind, ModelError
from ember.llm.provider import Decoded
from ember.llm.providers import build, gemini
from ember.llm.providers.http import Endpoint, Transport
from ember.llm.providers.http_provider import HttpProvider, Route
from ember.llm.providers.wire import Codec
from ember.llm.request import Request, Sampling
from ember.llm.result import FinishReason, TextDelta

KEY = "test-key-0123456789"


def _provider(kit, recorder, settings, respond):
    record = recorder(respond)
    return build(settings, kit.context(), transport=record.transport), record


def _completed(kit, name: str) -> str:
    """An event stream whose only event completes the response in ``name``."""
    event = {"type": "response.completed", "response": kit.json(name)}
    return f"event: response.completed\ndata: {json.dumps(event)}\n\n"


def test_openai_chat_url_auth_and_request_id(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("openai", "openai", key=KEY),
        lambda request: kit.sse_response(
            kit.text("openai/chat_stream_text.sse"), **{"x-request-id": "req_a"}
        ),
    )
    decoded = provider.send(
        kit.spec("openai_chat", wire_id="gpt-6-astra"), Request.from_prompt("hi")
    )
    request = record.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert record.last_body["model"] == "gpt-6-astra"
    assert decoded.request_id == "req_a"


def test_openai_routes_responses_models_by_catalog_api(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("openai", "openai", key=KEY),
        lambda request: kit.sse_response(_completed(kit, "openai/responses_reasoning.json")),
    )
    provider.send(kit.spec("openai_responses"), Request.from_prompt("hi"))
    assert record.requests[0].url.path == "/v1/responses"


def test_azure_v1_through_openai_compatible(kit, recorder) -> None:
    settings = kit.settings(
        "azure", "openai_compatible", key=KEY, base_url="https://myres.openai.azure.com/openai/v1"
    )
    provider, record = _provider(
        kit,
        recorder,
        settings,
        lambda request: kit.sse_response(
            kit.text("openai/chat_stream_text.sse"), **{"apim-request-id": "apim-1"}
        ),
    )
    spec = kit.spec("openai_chat", provider="azure", wire_id="my-gpt-deployment")
    decoded = provider.send(spec, Request.from_prompt("hi"))
    request = record.requests[0]
    # The v1 surface takes the deployment as the model and needs no api-version.
    assert str(request.url) == "https://myres.openai.azure.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert "api-key" not in request.headers
    assert record.last_body["model"] == "my-gpt-deployment"
    assert decoded.request_id == "apim-1"


def test_openai_compatible_key_is_optional_and_base_url_is_required(kit, recorder) -> None:
    settings = kit.settings(
        "local", "openai_compatible", key=None, base_url="http://localhost:8000/v1"
    )
    provider, record = _provider(
        kit,
        recorder,
        settings,
        lambda request: kit.sse_response(kit.text("openai/chat_stream_text.sse")),
    )
    provider.send(kit.spec("openai_chat", provider="local"), Request.from_prompt("hi"))
    assert "authorization" not in record.requests[0].headers
    with pytest.raises(ConfigError, match="base_url"):
        build(kit.settings("local", "openai_compatible", key=None), kit.context())


def test_anthropic_headers_and_path(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("anthropic", "anthropic", key=KEY),
        lambda request: kit.sse_response(
            kit.text("anthropic/stream_text.sse"), **{"request-id": "req_ant"}
        ),
    )
    spec = kit.spec("anthropic_messages", provider="anthropic", max_output_tokens=1024)
    decoded = provider.send(spec, Request.from_prompt("hi"))
    request = record.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["anthropic-version"] == "2026-10-21"
    assert "authorization" not in request.headers
    assert decoded.request_id == "req_ant"


def test_gemini_key_is_a_header_never_in_the_url(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("gemini", "gemini", key=KEY),
        lambda request: kit.sse_response(kit.text("gemini/stream_text.sse")),
    )
    spec = kit.spec("gemini", provider="gemini", wire_id="gemini-3.8-flash")
    provider.send(spec, Request.from_prompt("hi"))
    request = record.requests[0]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:streamGenerateContent?alt=sse"
    )
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)


def test_missing_key_names_the_variable(kit, monkeypatch) -> None:
    settings = kit.settings("openai", "openai", key=None, api_key_env="OPENAI_API_KEY")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build(settings, kit.context())
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-0123456789")
    assert build(settings, kit.context()).name == "openai"


@pytest.mark.parametrize("url", ["ftp://example.test/v1", "not a url", "https://"])
def test_base_url_must_be_http(kit, url) -> None:
    with pytest.raises(ConfigError, match="base_url"):
        build(kit.settings("openai", "openai", key=KEY, base_url=url), kit.context())


def test_model_on_another_wire_protocol_is_a_config_error(kit) -> None:
    provider = build(kit.settings("openai", "openai", key=KEY), kit.context())
    with pytest.raises(ConfigError, match="anthropic_messages"):
        provider.send(kit.spec("anthropic_messages"), Request.from_prompt("hi"))


def test_spec_for_another_provider_is_rejected(kit) -> None:
    provider = build(kit.settings("openai", "openai", key=KEY), kit.context())
    with pytest.raises(ValueError, match="not served"):
        provider.send(kit.spec("openai_chat", provider="azure"), Request.from_prompt("hi"))


def test_invalid_requests_fail_before_any_byte_is_sent(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("anthropic", "anthropic", key=KEY),
        lambda request: kit.sse_response(""),
    )
    spec = kit.spec("anthropic_messages", provider="anthropic", max_output_tokens=1024)
    with pytest.raises(ValueError):
        provider.send(spec, Request.from_prompt("hi", sampling=Sampling(temperature=0.3)))
    with pytest.raises(ValueError):
        provider.stream(spec, Request.from_prompt("hi", sampling=Sampling(temperature=0.3)))
    assert record.requests == []


def test_providers_never_retry(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("openai", "openai", key=KEY),
        lambda request: httpx.Response(503, json={"error": {"type": "server_overloaded"}}),
    )
    with pytest.raises(ModelError) as caught:
        provider.send(kit.spec("openai_chat"), Request.from_prompt("hi"))
    assert caught.value.kind is ErrorKind.OVERLOADED and caught.value.retryable
    assert len(record.requests) == 1


def test_codec_failures_become_model_errors_with_identity(kit, recorder) -> None:
    text = kit.text("openai/chat_stream_text.sse").replace(
        '"finish_reason":"stop"', '"finish_reason":"abort"'
    )
    provider, _ = _provider(
        kit,
        recorder,
        kit.settings("openai", "openai", key=KEY),
        lambda request: kit.sse_response(text, **{"x-request-id": "req_bad"}),
    )
    with pytest.raises(ModelError) as caught:
        provider.send(kit.spec("openai_chat", wire_id="gpt-6-astra"), Request.from_prompt("hi"))
    error = caught.value
    assert (error.kind, error.provider, error.model, error.request_id, error.retryable) == (
        ErrorKind.MALFORMED,
        "openai",
        "gpt-6-astra",
        "req_bad",
        False,
    )


def test_send_reads_the_stream_and_its_failures_delivered_nothing(kit, recorder) -> None:
    events = kit.text("anthropic/stream_text.sse").split("\n\n")
    text = "\n\n".join(events[:5]) + "\n\n" + kit.text("anthropic/stream_error_overloaded.sse")
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("anthropic", "anthropic", key=KEY),
        lambda request: kit.sse_response(text),
    )
    spec = kit.spec("anthropic_messages", provider="anthropic", max_output_tokens=1024)
    with pytest.raises(ModelError) as caught:
        provider.send(spec, Request.from_prompt("hi"))
    assert record.last_body["stream"] is True
    assert record.requests[0].headers["accept"] == "text/event-stream"
    # Two text deltas were read, but send() returns nothing before the end,
    # so the caller received nothing and the failure may be retried.
    assert (caught.value.kind, caught.value.before_first_byte, caught.value.retryable) == (
        ErrorKind.OVERLOADED,
        True,
        True,
    )


def test_unexpected_codec_exceptions_are_malformed(kit) -> None:
    def explode(spec, request, events, request_id):
        yield TextDelta("partial")
        raise KeyError("thinking")

    codec = Codec(encode=lambda spec, request: {}, decode_events=explode, classify=gemini.classify)
    endpoint = Endpoint("acme", "https://api.example.test/v1", {}, ())
    transport = Transport(
        endpoint,
        connect_timeout_s=1.0,
        idle_timeout_s=1.0,
        max_connections=1,
        transport=httpx.MockTransport(lambda request: kit.sse_response("data: {}\n\n")),
    )
    provider = HttpProvider("acme", transport, lambda spec: Route("x", codec))
    received = []
    with pytest.raises(ModelError) as caught:
        for item in provider.stream(kit.spec("gemini", provider="acme"), Request.from_prompt("hi")):
            received.append(item)
    assert received == [TextDelta("partial")]
    assert (caught.value.kind, caught.value.before_first_byte) == (ErrorKind.MALFORMED, False)
    assert isinstance(caught.value.__cause__, KeyError)


@pytest.mark.parametrize(("events_before_error", "delivered"), [(3, False), (4, True)])
def test_mid_stream_error_frame_says_whether_the_caller_received_anything(
    kit, recorder, events_before_error, delivered
) -> None:
    # message_start, content_block_start and ping yield nothing; the fourth
    # event is the first text delta.
    events = kit.text("anthropic/stream_text.sse").split("\n\n")
    text = "\n\n".join(events[:events_before_error]) + "\n\n"
    text += kit.text("anthropic/stream_error_overloaded.sse")
    provider, _ = _provider(
        kit,
        recorder,
        kit.settings("anthropic", "anthropic", key=KEY),
        lambda request: kit.sse_response(text, **{"request-id": "req_s"}),
    )
    spec = kit.spec("anthropic_messages", provider="anthropic", max_output_tokens=1024)
    received = []
    with pytest.raises(ModelError) as caught:
        for item in provider.stream(spec, Request.from_prompt("hi")):
            received.append(item)
    error = caught.value
    assert len(received) == int(delivered)
    assert (error.kind, error.request_id) == (ErrorKind.OVERLOADED, "req_s")
    assert error.before_first_byte is not delivered
    assert error.retryable is not delivered


def test_cut_before_any_item_is_retryable_transport(kit, recorder) -> None:
    text = kit.text("anthropic/stream_text.sse").split("event: content_block_delta")[0]
    provider, _ = _provider(
        kit,
        recorder,
        kit.settings("anthropic", "anthropic", key=KEY),
        lambda request: kit.sse_response(text),
    )
    spec = kit.spec("anthropic_messages", provider="anthropic", max_output_tokens=1024)
    with pytest.raises(ModelError) as caught:
        list(provider.stream(spec, Request.from_prompt("hi")))
    assert (caught.value.kind, caught.value.retryable) == (ErrorKind.TRANSPORT, True)


def test_stream_end_to_end(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("openai", "openai", key=KEY),
        lambda request: kit.sse_response(kit.text("openai/chat_stream_text.sse")),
    )
    items = list(provider.stream(kit.spec("openai_chat"), Request.from_prompt("hi")))
    assert record.requests[0].headers["accept"] == "text/event-stream"
    assert items[0] == TextDelta("Hello")
    assert isinstance(items[-1], Decoded) and items[-1].finish_reason is FinishReason.STOP


def test_importing_providers_loads_no_vendor_sdk(tmp_path) -> None:
    # The `import a.b.c as x` form walks attributes, so it also proves no
    # package attribute shadows a provider module.
    script = textwrap.dedent(
        """
        import sys
        import ember.llm.providers.gemini as gemini_codec
        import ember.llm.providers.openai_responses as responses_codec
        assert gemini_codec.CODEC is not responses_codec.CODEC
        vendors = {"openai", "anthropic", "google", "google.genai", "boto3"}
        loaded = sorted(vendors & set(sys.modules))
        assert not loaded, loaded
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_an_exception_the_consumer_throws_in_is_not_relabelled(kit, recorder) -> None:
    provider, _ = _provider(
        kit,
        recorder,
        kit.settings("openai", "openai", key=KEY),
        lambda request: kit.sse_response(kit.text("openai/chat_stream_text.sse")),
    )
    items = provider.stream(kit.spec("openai_chat"), Request.from_prompt("hi"))
    assert next(items) == TextDelta("Hello")
    with pytest.raises(ValueError, match="consumer"):
        items.throw(ValueError("consumer gave up"))


def test_gemini_invalid_key_reaches_the_caller_as_auth(kit, recorder) -> None:
    provider, record = _provider(
        kit,
        recorder,
        kit.settings("gemini", "gemini", key=KEY),
        lambda request: httpx.Response(400, json=kit.json("gemini/error_api_key_invalid.json")),
    )
    spec = kit.spec("gemini", provider="gemini", wire_id="gemini-3.8-flash")
    with pytest.raises(ModelError) as caught:
        provider.send(spec, Request.from_prompt("hi"))
    error = caught.value
    assert (error.kind, error.status, error.retryable) == (ErrorKind.AUTH, 400, False)
    assert "API_KEY_INVALID" in str(error) and "not valid" not in str(error)
    assert len(record.requests) == 1
