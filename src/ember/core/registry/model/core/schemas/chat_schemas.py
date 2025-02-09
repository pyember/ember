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
    """A universal chat response for Ember.

    Captures the generated output, raw provider output, and optional usage statistics.

    Attributes:
        data (str): The generated model output.
        raw_output (Any): The unprocessed data from the provider.
        usage (Optional[UsageStats]): Optional usage statistics associated with the response.
    """

    data: str
    raw_output: Any
    usage: Optional[UsageStats] = None
