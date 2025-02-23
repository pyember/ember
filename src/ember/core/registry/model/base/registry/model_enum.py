from enum import Enum
import logging
from typing import Dict, List, Type

logger: logging.Logger = logging.getLogger(__name__)


class OpenAIModelEnum(str, Enum):
    GPT_4O = "openai:gpt-4o"
    GPT_4O_MINI = "openai:gpt-4o-mini"
    O1 = "openai:o1"


class AnthropicModelEnum(str, Enum):
    CLAUDE_3_5_SONNET = "anthropic:claude-3.5-sonnet"

class GoogleModelEnum(str, Enum):
    GEMINI_1_5_PRO = "deepmind:gemini-1.5-pro"
    GEMINI_2_0_FLASH = "deepmind:gemini-2.0-flash"
    GEMINI_EXP_1206 = "deepmind:gemini-exp-1206"


ALL_PROVIDER_ENUMS: List[Type[Enum]] = [
    OpenAIModelEnum,
    AnthropicModelEnum,
    GoogleModelEnum,
]


def create_model_enum() -> Type[Enum]:
    members: Dict[str, str] = {}
    for provider_enum in ALL_PROVIDER_ENUMS:
        for model in provider_enum:
            if model.value in members.values():
                logger.warning("Duplicate model value detected: %s", model.value)
            members[model.name] = model.value
    return Enum("ModelEnum", members, type=str, module=__name__)


ModelEnum: Type[Enum] = create_model_enum()


def parse_model_str(model_str: str) -> str:
    """Parse and validate a model string against the aggregated ModelEnum.
    
    For unknown models, returns the original string to allow for dynamic/test models.
    """
    try:
        enum_member = ModelEnum(model_str)
        return enum_member.value
    except ValueError:
        raise ValueError(
            f"Invalid model ID '{model_str}'. Must match a known provider enum (e.g., 'openai:gpt-4o')."
        )
