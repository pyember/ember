import abc
from typing import Any, Optional

from pydantic import BaseModel, Field

from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)


class BaseChatParameters(BaseModel):
    """Base parameter model for LLM provider chat requests.

    This class defines the common parameters used across different language model providers,
    establishing a standardized interface for chat request configuration. Provider-specific
    implementations should extend this class to add or customize parameters according to
    their API requirements.
    
    Design principles:
    - Common parameters are standardized across providers
    - Sensible defaults reduce configuration burden
    - Validation built-in through Pydantic
    - Helper methods for common operations like prompt building
    
    Parameter semantics:
    - prompt: The core user input text to send to the model
    - context: Optional system context that provides additional information or instructions
    - temperature: Controls randomness/creativity (0.0 = deterministic, 2.0 = maximum randomness)
    - max_tokens: Optional limit on response length

    Usage:
    Provider-specific implementations should inherit from this class:
    ```python
    class AnthropicChatParameters(BaseChatParameters):
        top_k: Optional[int] = None
        top_p: Optional[float] = None
        # Additional Anthropic-specific parameters
    ```

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


class BaseProviderModel(abc.ABC):
    """Abstract base class defining the contract for all LLM provider implementations.
    
    This class establishes the core interface that all language model providers 
    (OpenAI, Anthropic, etc.) must implement to integrate with the Ember framework.
    It serves as the foundation of the provider abstraction layer, enabling a unified
    interface for working with different language models.
    
    Provider architecture:
    - Each provider must implement client creation and request handling
    - Models are instantiated with metadata through ModelInfo
    - Providers handle translating Ember's universal ChatRequest format into provider-specific formats
    - Responses are normalized back to Ember's ChatResponse format
    
    Lifecycle:
    1. Provider class is discovered and instantiated via ModelFactory 
    2. Provider creates its specific API client in create_client()
    3. Chat requests are processed through forward() or direct __call__
    
    Implementation requirements:
    - Subclasses must provide PROVIDER_NAME as a class attribute
    - Subclasses must implement create_client() and forward() methods
    - Client creation should handle authentication and configuration
    - Forward method must translate between Ember and provider-specific formats
    
    Usage example:
    ```python
    # Direct usage (prefer using ModelRegistry instead)
    model_info = ModelInfo(id="anthropic:claude-3", provider=ProviderInfo(name="anthropic"))
    provider = AnthropicProvider(model_info=model_info)
    response = provider("Tell me about the Ember framework")
    print(response.data)  # The model's response text
    ```
    """

    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize the provider model with the given model information.

        Args:
            model_info (ModelInfo): Metadata and configuration details for the model.
        """
        self.model_info: ModelInfo = model_info
        self.client: Any = self.create_client()

    @abc.abstractmethod
    def create_client(self) -> Any:
        """Create and configure the API client.

        Subclasses must override this method to initialize and return their API client.

        Returns:
            Any: A configured API client instance.
        """
        raise NotImplementedError("Subclasses must implement create_client")

    @abc.abstractmethod
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Process the chat request and return the corresponding response.

        Args:
            request (ChatRequest): The chat request containing the prompt and additional parameters.

        Returns:
            ChatResponse: The response generated by the provider.
        """
        raise NotImplementedError("Subclasses must implement forward")

    def __call__(self, prompt: str, **kwargs: Any) -> ChatResponse:
        """Allow the instance to be called as a function to process a prompt.

        This method constructs a ChatRequest using the prompt and keyword arguments,
        and then delegates the request processing to the forward() method.

        Args:
            prompt (str): The chat prompt to send.
            **kwargs (Any): Additional parameters to pass into the ChatRequest.

        Returns:
            ChatResponse: The response produced by processing the chat request.
        """
        chat_request: ChatRequest = ChatRequest(prompt=prompt, **kwargs)
        return self.forward(request=chat_request)
