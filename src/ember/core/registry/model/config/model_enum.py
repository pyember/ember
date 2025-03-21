import logging
from enum import Enum
from typing import Dict, List, Type

logger: logging.Logger = logging.getLogger(__name__)

"""
Model enums for the Ember framework.

This module defines standard model identifiers for commonly used models from
various providers including OpenAI, Anthropic, and Google/Deepmind. These 
model enums provide type-safe references to language model identifiers.

Note: The actual availability of models depends on API access and discovery.
Models defined in these enums represent a curated subset that are commonly used,
but the specific models available to your application will be determined at runtime
through API discovery.
"""


class OpenAIModelEnum(str, Enum):
    # GPT-4o models
    gpt_4o = "openai:gpt-4o"
    gpt_4o_mini = "openai:gpt-4o-mini"

    # GPT-4 models
    gpt_4 = "openai:gpt-4"
    gpt_4_turbo = "openai:gpt-4-turbo"

    # GPT-3.5 models
    gpt_3_5_turbo = "openai:gpt-3.5-turbo"

    # Reasoning models, which may require special access
    # Note: o1 support is pending
    o1 = "openai:o1-2024-12-17"


class AnthropicModelEnum(str, Enum):
    """
    Current Anthropic Claude models based on official documentation.

    This enum represents the currently available Claude models from Anthropic.
    Claude 3 models represent the base versions, with newer 3.5 and 3.7
    versions providing improved capabilities.
    """

    # Claude 3.7 models (latest generation)
    claude_3_7_sonnet = "anthropic:claude-3.7-sonnet"

    # Claude 3.5 models (using hyphen format as returned by API)
    claude_3_5_sonnet = "anthropic:claude-3-5-sonnet"
    claude_3_5_haiku = "anthropic:claude-3-5-haiku"

    # Claude 3 models (original generation)
    claude_3_opus = "anthropic:claude-3-opus"
    claude_3_haiku = "anthropic:claude-3-haiku"


class DeepmindModelEnum(str, Enum):
    # Gemini 1.5 models
    gemini_1_5_pro = "deepmind:gemini-1.5-pro"
    gemini_1_5_flash = "deepmind:gemini-1.5-flash"
    gemini_1_5_flash_8b = "deepmind:gemini-1.5-flash-8b"

    # Gemini 2.0 models
    gemini_2_0_flash = "deepmind:gemini-2.0-flash"
    gemini_2_0_flash_lite = "deepmind:gemini-2.0-flash-lite"
    gemini_2_0_pro = "deepmind:gemini-2.0-pro"


ALL_PROVIDER_ENUMS: List[Type[Enum]] = [
    OpenAIModelEnum,
    AnthropicModelEnum,
    DeepmindModelEnum,
]


def create_model_enum() -> Type[Enum]:
    """Create a unified ModelEnum from all provider-specific model enums.

    Returns:
        A new Enum type that includes all models from all providers.
    """
    members: Dict[str, str] = {}
    for provider_enum in ALL_PROVIDER_ENUMS:
        for model in provider_enum:
            if model.value in members.values():
                logger.warning("Duplicate model value detected: %s", model.value)
            members[model.name] = model.value
    # Cast to the expected return type
    return Enum("ModelEnum", members, type=str, module=__name__)  # type: ignore


ModelEnum: Type[Enum] = create_model_enum()


def parse_model_str(model_str: str) -> str:
    """Parse and validate a model string against the aggregated ModelEnum.

    Args:
        model_str: The model string to parse, e.g., "openai:gpt-4o"

    Returns:
        For known models, returns the standardized model identifier.
        For unknown models, returns the original string to allow for dynamic/test models.
    """
    try:
        enum_member = ModelEnum(model_str)
        return str(enum_member.value)  # Explicitly cast to str to satisfy mypy
    except ValueError:
        # Return the original string for unknown models
        # This enables dynamic registration of models not in the enum
        return model_str
