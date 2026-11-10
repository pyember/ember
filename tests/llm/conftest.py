"""Fixtures for model-call tests: an isolated config and a client on a fake provider."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ember import context
from ember.llm import Client
from ember.llm.fake import FakeProvider


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "config.yaml"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EMBER_CONFIG", str(path))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    context.reload()
    yield path
    context.reload()


@pytest.fixture
def fake() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def sleeps() -> list[float]:
    """Delays the client asked for between retries, in order."""
    return []


@pytest.fixture
def client(fake: FakeProvider, sleeps: list[float]) -> Iterator[Client]:
    # A jitter of 0.5 makes the backoff factor exactly 1.
    with Client(provider_factory=fake.factory, sleep=sleeps.append, jitter=lambda: 0.5) as c:
        yield c
