from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from ember.core.registry.model.core.schemas.usage import UsageStats


class ChatRequest(BaseModel):
    """A universal chat request for Ember.

    High-level API calls can utilize this model, and individual providers will adapt
    the request into their specific parameter classes.

    Attributes:
        prompt (str): The user prompt text.
        context (Optional[str]): Optional contextual information to guide the prompt.
        max_tokens (Optional[int]): Optional maximum number of tokens for the response.
        temperature (Optional[float]): Optional sampling temperature controlling randomness.
        provider_params (Dict[str, Any]): Optional provider-specific parameters.
    """

    prompt: str
    context: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    provider_params: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Standard response wrapper for chat models.

    Captures the generated output, raw provider output, and optional usage statistics.

    Attributes:
        data (str): The generated model output.
        raw_output (Any): The unprocessed data from the provider.
        usage (Optional[UsageStats]): Optional usage statistics associated with the response.
    """

    data: str
    raw_output: Any
    usage: Optional[UsageStats] = None


class BaseChatParameters(BaseModel):
    """Base chat parameters for provider-specific implementations.

    Providers should inherit from this class to manage common fields such as prompt,
    context, temperature, and token limitations.

    Attributes:
        prompt (str): The user prompt text.
        context (Optional[str]): Additional context to be prepended to the prompt.
        temperature (Optional[float]): Sampling temperature with a value between 0.0 and 2.0.
        max_tokens (Optional[int]): Optional maximum token count for responses.
    """

    prompt: str
    context: Optional[str] = None
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None

    def build_prompt(self) -> str:
        """Build the final prompt by combining context and the user prompt.

        Returns:
            str: The constructed prompt with context included when provided.
        """
        if self.context:
            return "{context}\n\n{prompt}".format(
                context=self.context, prompt=self.prompt
            )
        return self.prompt
