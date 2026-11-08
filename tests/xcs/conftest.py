"""Shares the oracle's fake model with the xcs unit tests and isolates config."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from ember import context

PROGRAMS = Path(__file__).resolve().parents[1] / "oracle" / "programs"
if str(PROGRAMS) not in sys.path:
    sys.path.insert(0, str(PROGRAMS))


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("EMBER_CONFIG", str(tmp_path / "config.yaml"))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    context.reload()
    yield
    context.reload()
