"""Unit tests for ModelEnum and parse_model_str functionality."""


from ember.core.registry.model.config.model_enum import (
    AnthropicModelEnum,
    DeepmindModelEnum,
    ModelEnum,
    OpenAIModelEnum,
    parse_model_str,
)


def test_model_enum_creation() -> None:
    """Test that ModelEnum combines models from all provider enums."""
    # Get all values from the ModelEnum
    all_values = [item.value for item in ModelEnum]

    # Check that at least one model from each provider exists in the enum
    assert any(
        val.startswith("openai:") for val in all_values
    ), "No OpenAI models found"
    assert any(
        val.startswith("anthropic:") for val in all_values
    ), "No Anthropic models found"
    assert any(
        val.startswith("deepmind:") for val in all_values
    ), "No Deepmind models found"


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
    assert OpenAIModelEnum.gpt_4o.value == "openai:gpt-4o"
    assert AnthropicModelEnum.claude_3_5_sonnet.value == "anthropic:claude-3-5-sonnet"
    assert DeepmindModelEnum.gemini_1_5_pro.value == "deepmind:gemini-1.5-pro"
