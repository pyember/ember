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
    """Universal chat request model that serves as the primary API for all provider requests.

    ChatRequest provides a unified interface for sending prompts to any supported language
    model provider. It encapsulates the common parameters used across all providers while
    also allowing provider-specific parameters through the provider_params field.
    
    This class is part of Ember's abstraction layer that enables seamless switching between
    different LLM providers without changing application code. Provider implementations
    translate this universal format into their specific API formats.
    
    Core design principles:
    - Provider-agnostic interface for consistent application code
    - Support for common parameters across all providers
    - Extensibility for provider-specific parameters
    - Type safety through Pydantic validation
    
    Flow:
    1. Application code creates a ChatRequest
    2. The request is passed to a model via ModelRegistry
    3. The provider implementation converts it to provider-specific format
    4. The provider sends the request and receives a response
    5. The response is converted back to a universal ChatResponse

    Example:
    ```python
    # Simple request
    request = ChatRequest(prompt="Explain quantum computing")
    
    # Request with additional parameters
    request = ChatRequest(
        prompt="Write a poem about spring",
        context="You are a professional poet",
        temperature=0.9,
        max_tokens=500,
        provider_params={"top_p": 0.95, "stop_sequences": ["---"]}
    )
    ```

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
    provider_params: Union[
        OpenAIProviderParams,
        AnthropicProviderParams,
        DeepmindProviderParams,
        ProviderParams,
    ] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Universal response model that standardizes outputs from all LLM providers.

    ChatResponse provides a consistent structure for handling responses from any language
    model provider integrated with Ember. It normalizes the diverse response formats from
    different providers into a unified interface that applications can rely on.
    
    Key features:
    - Standardized access to generated text across all providers
    - Preservation of raw provider output for advanced use cases
    - Optional usage tracking for cost management and monitoring
    - Pydantic validation for type safety
    
    Design patterns:
    - Adapter Pattern: Converts provider-specific responses to a universal format
    - Facade Pattern: Simplifies the complex provider responses into a clean interface
    - Data Transfer Object: Encapsulates all response data in a single value object
    
    Usage flow:
    1. Provider implementation receives raw response from API
    2. Provider converts the raw response into a ChatResponse
    3. Application code receives a consistent structure regardless of provider
    4. Application can access normalized text data or inspect raw provider output
    
    Example:
    ```python
    # Using a response
    response = model("Tell me about quantum computing")
    print(response.data)  # Access the generated text
    
    # Working with usage statistics
    if response.usage:
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")
        print(f"Estimated cost: ${response.usage.total_cost:.6f}")
    ```

    Attributes:
        data (str): The generated model output.
        raw_output (Any): The unprocessed data from the provider.
        usage (Optional[UsageStats]): Optional usage statistics associated with the response.
    """

    data: str
    raw_output: Any = None
    usage: Optional[UsageStats] = None
