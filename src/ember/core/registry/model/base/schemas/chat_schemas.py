from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union
from typing_extensions import TypedDict

from ember.core.registry.model.base.schemas.usage import UsageStats


class ProviderParams(TypedDict, total=False):
    """Base TypedDict for provider-specific parameters.
    
    This provides a common base for all provider parameter types.
    The total=False parameter makes all fields optional.
    """
    # Allow any string key with any value to maintain backward compatibility
    extra: Any


class OpenAIProviderParams(ProviderParams):
    """OpenAI-specific provider parameters."""
    stream: Optional[bool]
    stop: Optional[list[str]]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    top_p: Optional[float]
    seed: Optional[int]


class AnthropicProviderParams(ProviderParams):
    """Anthropic-specific provider parameters."""
    top_k: Optional[int]
    top_p: Optional[float]
    stop_sequences: Optional[list[str]]
    stream: Optional[bool]


class DeepmindProviderParams(ProviderParams):
    """Deepmind-specific provider parameters."""
    candidate_count: Optional[int]
    stop_sequences: Optional[list[str]]
    top_p: Optional[float]
    top_k: Optional[int]


class ChatRequest(BaseModel):
    """A universal chat request for Ember.

    High-level API calls can utilize this model, and individual providers will adapt
    the request into their specific parameter classes.

    Attributes:
        prompt (str): The user prompt text.
        context (Optional[str]): Optional contextual information to guide the prompt.
        max_tokens (Optional[int]): Optional maximum number of tokens for the response.
        temperature (Optional[float]): Optional sampling temperature controlling randomness.
        provider_params (ProviderParams): Optional provider-specific parameters.
    """

    prompt: str
    context: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    provider_params: Union[OpenAIProviderParams, AnthropicProviderParams, DeepmindProviderParams, ProviderParams] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """A universal chat response for Ember.

    Captures the generated output, raw provider output, and optional usage statistics.

    Attributes:
        data (str): The generated model output.
        raw_output (Any): The unprocessed data from the provider.
        usage (Optional[UsageStats]): Optional usage statistics associated with the response.
    """

    data: str
    raw_output: Any = None
    usage: Optional[UsageStats] = None
