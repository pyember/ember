from enum import Enum
import logging

class OpenAIModelEnum(str, Enum):
    """Models under the OpenAI provider."""
    GPT_4O = "openai:gpt-4o"
    GPT_4O_MINI = "openai:gpt-4o-mini"
    O1 = "openai:o1"

class AnthropicModelEnum(str, Enum):
    """Models under the Anthropic provider."""
    CLAUDE_3_5_SONNET = "anthropic:claude-3.5-sonnet"

class GoogleModelEnum(str, Enum):
    """Models under the Google/Gemini provider."""
    GEMINI_1_5_PRO = "google:gemini-1.5-pro"
    GEMINI_2_0_FLASH = "google:gemini-2.0-flash"
    GEMINI_EXP_1206 = "google:gemini-exp-1206"

# Optional aggregator for easy iteration
ALL_PROVIDER_ENUMS = [
    OpenAIModelEnum,
    AnthropicModelEnum,
    GoogleModelEnum,
]

def create_model_enum():
    """
    Dynamically build a str-based Enum that merges all known provider models
    into a single 'ModelEnum'.
    """
    members = {}
    for provider_enum in ALL_PROVIDER_ENUMS:
        for model in provider_enum:
            # model.name => e.g. "GPT_4O"
            # model.value => e.g. "openai:gpt-4o"
            members[model.name] = model.value
    return Enum("ModelEnum", members, type=str)

ModelEnum = create_model_enum()

def parse_model_str(model_str: str) -> str:
    """
    Global parse function to handle any recognized model string,
    searching across all known provider enum classes.
    Returns the same string if matched,
    or simply returns the original string if unrecognized.

    Example usage:
      parse_model_str("openai:gpt-4o") -> "openai:gpt-4o"
    """
    try:
        return ModelEnum(model_str).value
    except ValueError:
        logging.warning(
            f"{model_str} is not recognized among known provider enums. Returning raw string."
        )
        return model_str
