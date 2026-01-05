"""Regression tests for @jit stubbing and cache-miss execution behavior."""

from __future__ import annotations

from ember.xcs import jit

_COUNTER = {"value": 0}


def bump() -> int:
    _COUNTER["value"] += 1
    return _COUNTER["value"]


def kwonly_echo(*, y: int) -> int:
    return y


def test_jit_cache_miss_executes_once() -> None:
    _COUNTER["value"] = 0

    @jit
    def f() -> int:
        return bump()

    assert f() == 1
    assert _COUNTER["value"] == 1

    assert f() == 2
    assert _COUNTER["value"] == 2


def test_jit_kwonly_locals_are_materialized() -> None:
    @jit
    def f() -> int:
        z = 5
        return kwonly_echo(y=z)

    assert f() == 5
    assert f() == 5

