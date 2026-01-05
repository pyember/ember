"""Tests for the XCS JIT caching and config contract."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from ember.xcs.api.jit import BUILDER, jit
from ember.xcs.api.state import ENGINE
from ember.xcs.config import Config


def test_jit_avoids_retrace_for_identical_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_calls: list[tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]] = []
    original_trace = BUILDER.trace_with_result

    def trace_spy(func: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        trace_calls.append((args, tuple(sorted(kwargs.items()))))
        return original_trace(func, *args, **kwargs)

    monkeypatch.setattr(BUILDER, "trace_with_result", trace_spy)

    @jit
    def add(x: int, y: int) -> int:
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert len(trace_calls) == 1


def test_jit_retraces_when_cache_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_count = 0
    original_trace = BUILDER.trace_with_result

    def trace_spy(func: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        nonlocal trace_count
        trace_count += 1
        return original_trace(func, *args, **kwargs)

    monkeypatch.setattr(BUILDER, "trace_with_result", trace_spy)

    @jit
    def add(x: int, y: int) -> int:
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, _xcs_config=Config(cache=False)) == 3
    assert trace_count == 2


def test_jit_preserves_user_config_keyword() -> None:
    @jit
    def format_value(config: str, value: int) -> str:
        return f"{config}:{value}"

    assert format_value("mode", 5) == "mode:5"


def test_jit_warns_on_dual_config(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_count = 0
    original_trace = BUILDER.trace_with_result

    def trace_spy(func: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        nonlocal trace_count
        trace_count += 1
        return original_trace(func, *args, **kwargs)

    monkeypatch.setattr(BUILDER, "trace_with_result", trace_spy)

    @jit
    def add(x: int, y: int) -> int:
        return x + y

    with pytest.warns(UserWarning):
        assert add(1, 2, config={"parallel": False}, _xcs_config=Config()) == 3
    assert trace_count == 1


def test_typeerror_fallback_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original_execute = ENGINE.execute
    original_trace = BUILDER.trace_with_result

    exec_calls = {"count": 0}
    trace_calls = {"count": 0}

    def execute_stub(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        exec_calls["count"] += 1
        raise TypeError("missing a required argument: y")

    def trace_spy(func: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        trace_calls["count"] += 1
        return original_trace(func, *args, **kwargs)

    monkeypatch.setattr(ENGINE, "execute", execute_stub)
    monkeypatch.setattr(BUILDER, "trace_with_result", trace_spy)

    try:

        @jit
        def add(x: int, y: int) -> int:
            return x + y

        with caplog.at_level(logging.DEBUG):
            assert add(1, 2) == 3
        assert exec_calls["count"] == 0
        assert trace_calls["count"] == 1

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            assert add(1, 2) == 3
        assert exec_calls["count"] == 1
        assert trace_calls["count"] == 1
        assert any(
            "Graph execution raised TypeError" in record.message for record in caplog.records
        )

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            assert add(1, 2) == 3
        assert exec_calls["count"] == 1
        assert trace_calls["count"] == 2
        assert not any(
            "Graph execution raised TypeError" in record.message for record in caplog.records
        )
    finally:
        monkeypatch.setattr(ENGINE, "execute", original_execute)
        monkeypatch.setattr(BUILDER, "trace_with_result", original_trace)


def test_typeerror_fallback_surfaces_direct_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_execute = ENGINE.execute

    def execute_stub(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise TypeError("engine failure")

    monkeypatch.setattr(ENGINE, "execute", execute_stub)

    try:

        @jit
        def maybe_fail() -> int:
            if should_fail["value"]:
                raise TypeError("direct failure")
            return 1

        should_fail = {"value": False}
        assert maybe_fail() == 1

        should_fail["value"] = True
        with pytest.raises(TypeError) as exc_info:
            maybe_fail()
        assert str(exc_info.value) == "direct failure"
        assert exc_info.value.__cause__ is not None
        assert str(exc_info.value.__cause__) == "engine failure"
    finally:
        monkeypatch.setattr(ENGINE, "execute", original_execute)


def test_jit_skips_cache_when_cache_key_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_count = 0
    original_trace = BUILDER.trace_with_result

    def trace_spy(func: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        nonlocal trace_count
        trace_count += 1
        return original_trace(func, *args, **kwargs)

    monkeypatch.setattr(BUILDER, "trace_with_result", trace_spy)

    class Token:
        pass

    @jit
    def add(x: int, token: Token) -> int:
        return x + 1

    token = Token()
    assert add(1, token) == 2
    assert add(1, token) == 2
    assert trace_count == 2

    stats = add.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 2
    assert stats["cache_key_failures"] == 2


def test_jit_counts_tensor_classify_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    jit_module = importlib.import_module("ember.xcs.api.jit")

    def classify_stub(_graph: Any) -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(jit_module, "_graph_is_pure_tensor", classify_stub)

    @jit
    def add(x: int, y: int) -> int:
        return x + y

    assert add(1, 2) == 3
    stats = add.stats()
    assert stats["pure_tensor_classify_failures"] == 1
