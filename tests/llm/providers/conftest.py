"""Shared pieces for provider tests: wire fixtures, model specs and contexts.

No test here touches the network or a real key: providers get an
``httpx.MockTransport`` and keys are literals.
"""

import json
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx
import pytest

from ember.context import Context, ProviderSettings, Secret
from ember.llm.providers.sse import SseEvent, parse
from ember.llm.ref import ModelRef
from ember.llm.request import WireApi
from ember.llm.spec import ModelSpec

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keeps every provider test away from the owner's keys and config."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EMBER_CONFIG", str(tmp_path / "config.yaml"))


class Kit:
    """Helpers every provider test uses, handed out by the ``kit`` fixture."""

    @staticmethod
    def text(name: str) -> str:
        """Reads a fixture file as text."""
        return (FIXTURES / name).read_text(encoding="utf-8")

    @staticmethod
    def json(name: str) -> Any:
        """Reads a JSON fixture; tests may change the returned copy."""
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    @staticmethod
    def events(name: str) -> list[SseEvent]:
        """Parses an SSE fixture into events."""
        return list(parse([(FIXTURES / name).read_bytes()]))

    @staticmethod
    def sse(*payloads: object) -> list[SseEvent]:
        """Builds events whose data are the JSON of ``payloads``; strings pass as-is."""
        return [SseEvent("message", p if isinstance(p, str) else json.dumps(p)) for p in payloads]

    @staticmethod
    def spec(
        api: WireApi,
        *,
        provider: str = "openai",
        wire_id: str = "test-model",
        max_output_tokens: int | None = None,
    ) -> ModelSpec:
        """Returns a catalog-like spec for ``provider/wire_id`` speaking ``api``."""
        return ModelSpec(
            ref=ModelRef(provider, wire_id),
            wire_id=wire_id,
            api=api,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def context() -> Context:
        """Returns a context with the default liveness bounds and no providers."""
        return Context(providers=MappingProxyType({}))

    @staticmethod
    def settings(
        name: str,
        kind: str,
        *,
        key: str | None = "test-key-0123456789",
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> ProviderSettings:
        """Returns provider settings holding a literal test key."""
        return ProviderSettings(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            api_key_env=api_key_env,
            api_key=None if key is None else Secret(key),
            base_url=base_url,
        )

    @staticmethod
    def json_response(body: object, status: int = 200, **headers: str) -> httpx.Response:
        """Returns a JSON response with ``headers``."""
        return httpx.Response(status, json=body, headers=headers)

    @staticmethod
    def sse_response(text: str, **headers: str) -> httpx.Response:
        """Returns an event-stream response carrying ``text``."""
        return httpx.Response(
            200,
            content=text.encode("utf-8"),
            headers={"content-type": "text/event-stream", **headers},
        )


class Recorder:
    """A mock transport that records requests and answers from a handler."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        return self._respond(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def last_body(self) -> Any:
        return json.loads(self.requests[-1].content)


@pytest.fixture
def kit() -> Kit:
    """Shared helpers: fixture files, specs, settings and canned responses."""
    return Kit()


@pytest.fixture
def recorder() -> Callable[[Callable[[httpx.Request], httpx.Response]], Recorder]:
    """Returns a factory for request-recording mock transports."""
    return Recorder
