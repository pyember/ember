"""Unit tests for ModelEnum and parse_model_str functionality."""

import pytest

from ember.core.registry.model.config.model_enum import (
    AnthropicModelEnum,
    DeepmindModelEnum,
    ModelEnum,
    OpenAIModelEnum,
    parse_model_str,
)


def test_model_enum_creation() -> None:
    """Test that ModelEnum combines models from all provider enums."""
    # Verify at least one model from each provider is in the aggregated enum
    assert OpenAIModelEnum.GPT_4O.value in [item.value for item in ModelEnum]
    assert AnthropicModelEnum.CLAUDE_3_5_SONNET.value in [
        item.value for item in ModelEnum
    ]
    assert DeepmindModelEnum.GEMINI_1_5_PRO.value in [item.value for item in ModelEnum]


def test_known_model_enum() -> None:
    """Test that a known model string is parsed correctly via ModelEnum."""
    value = parse_model_str("openai:gpt-4o")
    assert value == "openai:gpt-4o"


def test_unknown_model_enum() -> None:
    """Test that an unknown model string is returned as-is."""
    # The implementation now returns the original string instead of raising ValueError
    value = parse_model_str("unknown:model")
    assert value == "unknown:model"


def test_provider_enum_values() -> None:
    """Test that provider-specific enums have expected values."""
    assert OpenAIModelEnum.GPT_4O.value == "openai:gpt-4o"
    assert AnthropicModelEnum.CLAUDE_3_5_SONNET.value == "anthropic:claude-3.5-sonnet"
    assert DeepmindModelEnum.GEMINI_1_5_PRO.value == "deepmind:gemini-1.5-pro"
