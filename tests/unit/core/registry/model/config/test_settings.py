#!/usr/bin/env python3
"""Unit tests for configuration functions in settings.py.

This module tests the deep_merge and resolve_env_vars functions.
"""

from typing import Any, Dict

import os
import pytest

from src.ember.core.registry.model.config.settings import deep_merge, resolve_env_vars


def test_deep_merge_dicts() -> None:
    """Test deep_merge with nested dictionaries."""
    base: Dict[str, Any] = {"a": 1, "b": {"x": 10, "y": 20}}
    override: Dict[str, Any] = {"b": {"y": 200, "z": 300}, "c": 3}
    expected: Dict[str, Any] = {"a": 1, "b": {"x": 10, "y": 200, "z": 300}, "c": 3}
    result = deep_merge(base=base, override=override)
    assert result == expected


def test_deep_merge_lists() -> None:
    """Test deep_merge with lists."""
    base = [1, 2, 3]
    override = [4, 5]
    expected = [1, 2, 3, 4, 5]
    result = deep_merge(base=base, override=override)
    assert result == expected


def test_deep_merge_mixed() -> None:
    """Test deep_merge with mixed types (dict and list)."""
    base = {"a": [1, 2], "b": "old"}
    override = {"a": [3], "b": "new"}
    expected = {"a": [1, 2, 3], "b": "new"}
    result = deep_merge(base=base, override=override)
    assert result == expected


def test_resolve_env_vars_with_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variable placeholders are correctly resolved."""
    monkeypatch.setenv("TEST_VAR", "resolved_value")
    data: Any = {"key": "${TEST_VAR}", "unchanged": "no_placeholder"}
    expected = {"key": "resolved_value", "unchanged": "no_placeholder"}
    result = resolve_env_vars(data=data)
    assert result == expected


def test_resolve_env_vars_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that resolve_env_vars works for nested data structures."""
    monkeypatch.setenv("NESTED_VAR", "nested")
    data = {"outer": {"inner": "${NESTED_VAR}"}, "list": ["${NESTED_VAR}", "static"]}
    expected = {"outer": {"inner": "nested"}, "list": ["nested", "static"]}
    result = resolve_env_vars(data=data)
    assert result == expected