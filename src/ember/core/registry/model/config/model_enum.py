from enum import Enum
import logging
from typing import Dict, List, Type

logger: logging.Logger = logging.getLogger(__name__)


class OpenAIModelEnum(str, Enum):
    """Enumeration of OpenAI provider models.

    Attributes:
        GPT_4O: Identifier for the GPT-4O model.
        GPT_4O_MINI: Identifier for the GPT-4O Mini model.
        O1: Identifier for the O1 model.
    """

    GPT_4O: str = "openai:gpt-4o"
    GPT_4O_MINI: str = "openai:gpt-4o-mini"
    O1: str = "openai:o1"


class AnthropicModelEnum(str, Enum):
    """Enumeration of Anthropic provider models.

    Attributes:
        CLAUDE_3_5_SONNET: Identifier for the Claude 3.5 Sonnet model.
    """

    CLAUDE_3_5_SONNET: str = "anthropic:claude-3.5-sonnet"


class GoogleModelEnum(str, Enum):
    """Enumeration of Google/Gemini provider models.

    Attributes:
        GEMINI_1_5_PRO: Identifier for the Gemini 1.5 Pro model.
        GEMINI_2_0_FLASH: Identifier for the Gemini 2.0 Flash model.
        GEMINI_EXP_1206: Identifier for the Gemini EXP 1206 model.
    """

    GEMINI_1_5_PRO: str = "google:gemini-1.5-pro"
    GEMINI_2_0_FLASH: str = "google:gemini-2.0-flash"
    GEMINI_EXP_1206: str = "google:gemini-exp-1206"


ALL_PROVIDER_ENUMS: List[Type[Enum]] = [
    OpenAIModelEnum,
    AnthropicModelEnum,
    GoogleModelEnum,
]


def create_model_enum() -> Type[Enum]:
    """Dynamically create an aggregated Enum for all provider models.

    Combines model identifiers from each provider-specific Enum into a single Enum
    named "ModelEnum". Each member's name corresponds to the model name and its value
    is the standardized model identifier.

    Returns:
        Type[Enum]: The dynamically constructed ModelEnum.
    """
    members: Dict[str, str] = {
        model.name: model.value
        for provider_enum in ALL_PROVIDER_ENUMS
        for model in provider_enum
    }
    enum_name: str = "ModelEnum"
    # Create the merged Enum using explicit keyword arguments for clarity.
    model_enum: Type[Enum] = Enum(
        enum_name,
        members,
        type=str,
        module=__name__,
    )
    return model_enum


ModelEnum: Type[Enum] = create_model_enum()


def parse_model_str(model_str: str) -> str:
    """Parse and validate a model string against known provider models.

    Searches the aggregated ModelEnum for a member matching the input string.
    If found, returns the standardized model identifier; otherwise, logs a warning
    and returns the original string.

    Args:
        model_str (str): The model identifier string to validate.

    Returns:
        str: The validated model identifier if recognized, else the original string.

    Example:
        >>> parse_model_str("openai:gpt-4o")
        'openai:gpt-4o'
    """
    try:
        enum_member: Enum = ModelEnum(model_str)
        return enum_member.value
    except ValueError:
        logger.warning(
            "%s is not recognized among known provider enums. Returning raw string.",
            model_str,
        )
        return model_str
