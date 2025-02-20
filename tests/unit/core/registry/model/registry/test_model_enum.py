#!/usr/bin/env python3
"""Unit tests for ModelEnum and parse_model_str functionality.
"""

import pytest

from src.ember.core.registry.model.registry.model_enum import parse_model_str


def test_known_model_enum() -> None:
    """Test that a known model string is parsed correctly via ModelEnum."""
    value = parse_model_str("openai:gpt-4o")
    assert value == "openai:gpt-4o"


def test_unknown_model_enum() -> None:
    """Test that an unknown model string returns itself."""
    value = parse_model_str("unknown:model")
    assert value == "unknown:model"
