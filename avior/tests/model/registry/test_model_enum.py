import pytest
from src.avior.registry.model.registry.model_enum import (
    parse_model_str,
    OpenAIModelEnum,
    AnthropicModelEnum,
    GoogleModelEnum,
    WatsonxModelEnum,
)


def test_openai_enum_values():
    assert OpenAIModelEnum.GPT_4.value == "openai:gpt-4"


def test_anthropic_enum_values():
    assert AnthropicModelEnum.CLAUDE_2.value == "anthropic:claude-2"


def test_parse_model_str_success():
    val = parse_model_str("openai:gpt-4")
    assert val == "openai:gpt-4"


def test_parse_model_str_failure():
    with pytest.raises(ValueError):
        parse_model_str("nonexistent:some-model")
