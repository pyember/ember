"""Provider implementations, one codec module per wire protocol.

`build` is the only entry point the client uses. Each provider kind is an
`HttpProvider` with its own endpoint, authentication and routing, and every
call streams:

- ``openai``: Chat Completions or Responses, chosen by the model's catalog
  ``api``.
- ``openai_compatible``: Chat Completions at a configured ``base_url``; the
  key is optional. The server must support streaming (``stream: true`` with
  a ``text/event-stream`` response), because every call streams; one that
  answers with plain JSON fails with a `MALFORMED` error saying so. Azure
  OpenAI works through it with the
  ``https://<resource>.openai.azure.com/openai/v1`` base URL, which accepts
  the key as a bearer token and needs no ``api-version``.
- ``anthropic``: Messages, at API version
  `anthropic_messages.API_VERSION`.
- ``gemini``: ``streamGenerateContent``.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from urllib.parse import quote

import httpx

from ember.context import Context, ProviderSettings
from ember.errors import ConfigError
from ember.llm.provider import Provider
from ember.llm.providers import anthropic_messages, gemini, openai_chat, openai_responses
from ember.llm.providers.http import Endpoint, Transport
from ember.llm.providers.http_provider import HttpProvider, Route, Router
from ember.llm.spec import ModelSpec

__all__ = ["build"]

_OPENAI_URL = "https://api.openai.com/v1"
_ANTHROPIC_URL = "https://api.anthropic.com/v1"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"


def build(
    settings: ProviderSettings, ctx: Context, *, transport: httpx.BaseTransport | None = None
) -> Provider:
    """Constructs the provider described by ``settings``.

    The returned provider owns its HTTP client; the caller closes it.

    Args:
        settings: Which provider, where it lives, and its credentials.
        ctx: Supplies the connect and idle timeouts and the connection limit.
        transport: Replaces the network, for tests.

    Raises:
        ConfigError: A required key or base URL is missing, or the base URL
            is not an http(s) URL.
    """
    endpoint, router = _KINDS[settings.kind](settings)
    connection = Transport(
        endpoint,
        connect_timeout_s=ctx.connect_timeout_s,
        idle_timeout_s=ctx.idle_timeout_s,
        max_connections=ctx.max_in_flight_per_provider,
        transport=transport,
    )
    return HttpProvider(settings.name, connection, router)


def _openai(settings: ProviderSettings) -> tuple[Endpoint, Router]:
    key = _required_key(settings)
    endpoint = Endpoint(
        provider=settings.name,
        base_url=_base_url(settings, _OPENAI_URL),
        headers={"authorization": f"Bearer {key}"},
        request_id_headers=("x-request-id",),
    )

    def route(spec: ModelSpec) -> Route:
        if spec.api == "openai_chat":
            return Route("chat/completions", openai_chat.CODEC)
        if spec.api == "openai_responses":
            return Route("responses", openai_responses.CODEC)
        raise _unspoken(settings, spec)

    return endpoint, route


def _openai_compatible(settings: ProviderSettings) -> tuple[Endpoint, Router]:
    if settings.base_url is None:
        raise ConfigError(f"provider {settings.name!r} is openai_compatible and needs base_url")
    key = settings.credential()
    endpoint = Endpoint(
        provider=settings.name,
        base_url=_base_url(settings, settings.base_url),
        headers={} if key is None else {"authorization": f"Bearer {key.reveal()}"},
        request_id_headers=("x-request-id", "apim-request-id"),
    )

    def route(spec: ModelSpec) -> Route:
        if spec.api != "openai_chat":
            raise _unspoken(settings, spec)
        return Route("chat/completions", openai_chat.CODEC)

    return endpoint, route


def _anthropic(settings: ProviderSettings) -> tuple[Endpoint, Router]:
    endpoint = Endpoint(
        provider=settings.name,
        base_url=_base_url(settings, _ANTHROPIC_URL),
        headers={
            "x-api-key": _required_key(settings),
            "anthropic-version": anthropic_messages.API_VERSION,
        },
        request_id_headers=("request-id",),
    )

    def route(spec: ModelSpec) -> Route:
        if spec.api != "anthropic_messages":
            raise _unspoken(settings, spec)
        return Route("messages", anthropic_messages.CODEC)

    return endpoint, route


def _gemini(settings: ProviderSettings) -> tuple[Endpoint, Router]:
    endpoint = Endpoint(
        provider=settings.name,
        base_url=_base_url(settings, _GEMINI_URL),
        headers={"x-goog-api-key": _required_key(settings)},
        request_id_headers=(),
    )

    def route(spec: ModelSpec) -> Route:
        if spec.api != "gemini":
            raise _unspoken(settings, spec)
        model = f"models/{quote(spec.wire_id, safe='')}"
        return Route(f"{model}:streamGenerateContent", gemini.CODEC, {"alt": "sse"})

    return endpoint, route


_KINDS: Mapping[str, Callable[[ProviderSettings], tuple[Endpoint, Router]]] = MappingProxyType(
    {
        "openai": _openai,
        "openai_compatible": _openai_compatible,
        "anthropic": _anthropic,
        "gemini": _gemini,
    }
)


def _required_key(settings: ProviderSettings) -> str:
    key = settings.credential()
    if key is None:
        where = f"set {settings.api_key_env} or " if settings.api_key_env else ""
        raise ConfigError(
            f"provider {settings.name!r} has no API key; {where}add api_key to its config"
        )
    return key.reveal()


def _base_url(settings: ProviderSettings, default: str) -> str:
    url = settings.base_url or default
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        raise ConfigError(f"provider {settings.name!r} base_url is not a valid URL") from None
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise ConfigError(f"provider {settings.name!r} base_url must be an http(s) URL")
    return url


def _unspoken(settings: ProviderSettings, spec: ModelSpec) -> ConfigError:
    return ConfigError(
        f"provider {settings.name!r} ({settings.kind}) cannot call {spec.ref}, "
        f"whose catalog entry uses the {spec.api} wire protocol"
    )
