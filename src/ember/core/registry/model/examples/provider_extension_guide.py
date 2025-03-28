"""Provider Extension Guide for Ember: Adding Text Completion and Embeddings Models.

This module demonstrates how to extend Ember's model registry system to support
new model types beyond chat models. It provides a complete example implementation for:

1. Adding text completion models (traditional completion API)
2. Adding embedding models support
3. Extending the provider system with multiple model type capabilities

The guide follows Ember's architectural principles:
- Minimalism and clean interfaces
- Strong typing with explicit annotations
- Extensibility through well-defined abstractions
- Backward compatibility with existing systems
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, TypeVar, Union, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Protocol, TypedDict

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.plugin_system import provider

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PART 1: Schema Definitions for Text Completion and Embeddings
# -----------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """Universal text completion request model.

    Similar to ChatRequest but designed for single-turn text completion.
    Used for traditional completion models that predate chat-oriented models.

    Attributes:
        prompt: The text prompt to complete.
        max_tokens: Optional maximum number of tokens to generate.
        temperature: Optional sampling temperature controlling randomness.
        stop_sequences: Optional list of sequences that signal the end of generation.
        provider_params: Provider-specific parameters as a flexible dictionary.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stop_sequences: Optional[List[str]] = None
    provider_params: ProviderParams = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Universal text completion response model.

    Standardizes the response format for text completion models.

    Attributes:
        text: The generated completion text.
        raw_output: The unprocessed provider-specific response data.
        usage: Optional usage statistics for token counting and cost tracking.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    text: str
    raw_output: Any = None
    usage: Optional[UsageStats] = None


class EmbeddingRequest(BaseModel):
    """Request model for generating vector embeddings from text.

    Used to generate semantic vector representations that capture the meaning
    of input text, suitable for similarity comparisons, clustering, and search.

    Attributes:
        input: Text input(s) to embed - can be a single string or list of strings.
        model: Optional specific embedding model to use when the provider has multiple.
        provider_params: Provider-specific parameters as a flexible dictionary.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    input: Union[str, List[str]]
    model: Optional[str] = None
    provider_params: ProviderParams = Field(default_factory=dict)

    @field_validator("input")
    def validate_input(cls, value: Union[str, List[str]]) -> Union[str, List[str]]:
        """Validating the input text is not empty.

        Args:
            value: The input text(s) to validate.

        Returns:
            The validated input value.

        Raises:
            ValueError: If input is empty string or empty list.
        """
        if isinstance(value, str) and not value.strip():
            raise ValueError("Input text cannot be empty")
        if isinstance(value, list) and (
            len(value) == 0 or all(not t.strip() for t in value)
        ):
            raise ValueError("Input list cannot be empty or contain only empty strings")
        return value


class EmbeddingResponse(BaseModel):
    """Response model containing vector embeddings.

    Contains numerical vector representations of input text that capture semantic meaning.

    Attributes:
        embeddings: Vector representation(s) of the input text(s).
        model: Name of the embedding model used.
        dimensions: The dimensionality of the embedding vectors.
        raw_output: The unprocessed provider-specific response data.
        usage: Optional usage statistics for token counting and cost tracking.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    embeddings: Union[List[float], List[List[float]]]
    model: str
    dimensions: int
    raw_output: Any = None
    usage: Optional[UsageStats] = None


# Type variable for implementation-specific typing
ModelT = TypeVar("ModelT", bound="CapabilityModel")


# -----------------------------------------------------------------------------
# PART 2: Protocol Definitions for Model Capabilities
# -----------------------------------------------------------------------------


class TextCompletionCapable(Protocol):
    """Protocol defining the interface for text completion models.

    Provider implementations supporting text completion should implement this protocol.
    """

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Processing a text completion request.

        Args:
            request: The text completion request.

        Returns:
            The text completion response.
        """
        ...

    def complete_text(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Convenience method for simple text completion.

        Args:
            prompt: The text to complete.
            **kwargs: Additional parameters for the completion request.

        Returns:
            The text completion response.
        """
        ...


class EmbeddingCapable(Protocol):
    """Protocol defining the interface for embedding models.

    Provider implementations supporting embeddings should implement this protocol.
    """

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generating embeddings for the input text(s).

        Args:
            request: The embedding request.

        Returns:
            The embedding response containing vector representations.
        """
        ...

    def embed_text(
        self, input_text: Union[str, List[str]], **kwargs: Any
    ) -> EmbeddingResponse:
        """Convenience method for simple embedding generation.

        Args:
            input_text: The text(s) to embed.
            **kwargs: Additional parameters for the embedding request.

        Returns:
            The embedding response with vector representations.
        """
        ...


# Base class for capability-aware models
class CapabilityModel(BaseProviderModel):
    """Extended base provider model with capability flags.

    This class extends BaseProviderModel with explicit capability tracking
    to allow runtime capability detection for different model types.

    Attributes:
        CAPABILITIES: Class variable mapping capability names to support flags.
    """

    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        "chat": True,
        "completion": False,
        "embedding": False,
    }


# -----------------------------------------------------------------------------
# PART 3: Extended Provider Base Classes
# -----------------------------------------------------------------------------


class TextCompletionProviderModel(CapabilityModel, TextCompletionCapable):
    """Base class for text completion model providers.

    Extends the BaseProviderModel to support text completion capabilities.
    Providers supporting text completion should inherit from this class.
    """

    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        "chat": True,
        "completion": True,
        "embedding": False,
    }

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Processing a text completion request.

        Args:
            request: The text completion request.

        Returns:
            The text completion response.

        Raises:
            NotImplementedError: If the provider has not implemented this capability.
        """
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support text completion"
        )

    def complete_text(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Convenience method for text completion.

        Creates a CompletionRequest from the prompt and additional parameters,
        then delegates to the complete() method for processing.

        Args:
            prompt: The text to complete.
            **kwargs: Additional parameters for the completion request.

        Returns:
            The text completion response.
        """
        request = CompletionRequest(prompt=prompt, **kwargs)
        return self.complete(request=request)


class EmbeddingProviderModel(CapabilityModel, EmbeddingCapable):
    """Base class for embedding model providers.

    Extends the BaseProviderModel to support embedding capabilities.
    Providers supporting embeddings should inherit from this class.
    """

    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        "chat": True,
        "completion": False,
        "embedding": True,
    }

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generating embeddings for the input text(s).

        Args:
            request: The embedding request.

        Returns:
            The embedding response containing vector representations.

        Raises:
            NotImplementedError: If the provider has not implemented this capability.
        """
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support embeddings"
        )

    def embed_text(
        self, input_text: Union[str, List[str]], **kwargs: Any
    ) -> EmbeddingResponse:
        """Convenience method for generating embeddings.

        Creates an EmbeddingRequest from the input text and additional parameters,
        then delegates to the embed() method for processing.

        Args:
            input_text: The text(s) to embed.
            **kwargs: Additional parameters for the embedding request.

        Returns:
            The embedding response with vector representations.
        """
        request = EmbeddingRequest(input=input_text, **kwargs)
        return self.embed(request=request)


# -----------------------------------------------------------------------------
# PART 4: Complete Provider Implementation (OpenAI Example)
# -----------------------------------------------------------------------------


class OpenAICompletionParams(TypedDict, total=False):
    """OpenAI-specific parameters for text completion.

    Additional parameters for fine-tuning OpenAI text completion API requests.
    """

    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    best_of: Optional[int]
    logit_bias: Optional[Dict[str, float]]
    suffix: Optional[str]
    echo: Optional[bool]


class OpenAICompletionParameters(BaseModel):
    """Parameter conversion for OpenAI text completion requests.

    Handles parameter validation and conversion between Ember's universal format
    and OpenAI's specific API requirements.

    Attributes:
        prompt: The text prompt to complete.
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness (0.0-2.0).
        stop_sequences: Sequences that signal end of generation.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    prompt: str
    max_tokens: Optional[int] = Field(default=50)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stop_sequences: Optional[List[str]] = None

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """Converting parameters to OpenAI API format.

        Returns:
            Dictionary of parameters for the OpenAI API.
        """
        kwargs: Dict[str, Any] = {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.stop_sequences:
            kwargs["stop"] = self.stop_sequences

        return kwargs


@provider("OpenAIExtended")
class OpenAIExtendedModel(TextCompletionProviderModel, EmbeddingProviderModel):
    """Extended OpenAI provider supporting chat, text completion, and embeddings.

    This example demonstrates how to implement a provider that supports multiple
    model types through capability interfaces.

    Attributes:
        PROVIDER_NAME: Provider name for registration with the plugin system.
        CAPABILITIES: Capability flags showing supported model types.
    """

    PROVIDER_NAME: ClassVar[str] = "OpenAIExtended"
    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        "chat": True,
        "completion": True,
        "embedding": True,
    }

    def create_client(self) -> Any:
        """Creating and configuring the OpenAI client.

        Retrieves the API key from the model information and configures the client.

        Returns:
            The configured OpenAI client.

        Raises:
            ProviderAPIError: If API key is missing or invalid.
        """
        import openai

        api_key: Optional[str] = self.model_info.get_api_key()
        if not api_key:
            raise ProviderAPIError("OpenAI API key is missing or invalid.")

        openai.api_key = api_key
        return openai

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Processing a chat request (implementing BaseProviderModel).

        This method provides the standard chat functionality required by
        the BaseProviderModel interface.

        Args:
            request: Chat request to process.

        Returns:
            Chat response from the model.

        Raises:
            InvalidPromptError: If prompt is empty.
            ProviderAPIError: For unexpected errors during API calls.
        """
        # Implementation would match OpenAIModel's forward method
        # This is a simplified placeholder
        if not request.prompt:
            raise InvalidPromptError("OpenAI prompt cannot be empty.")

        # Implementation details would mirror the standard OpenAIModel
        # Return placeholder
        return ChatResponse(data="Chat implementation placeholder")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Processing a text completion request.

        Implements text completion capabilities using the OpenAI completions API.

        Args:
            request: Text completion request.

        Returns:
            Completion response from the model.

        Raises:
            InvalidPromptError: If prompt is empty.
            ProviderAPIError: For unexpected errors during API calls.
        """
        if not request.prompt:
            raise InvalidPromptError("OpenAI completion prompt cannot be empty.")

        logger.info(
            "OpenAI completion invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": self.model_info.name,
                "prompt_length": len(request.prompt),
            },
        )

        # Convert universal parameters to OpenAI format
        openai_parameters = OpenAICompletionParameters(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_sequences=request.stop_sequences,
        )
        openai_kwargs = openai_parameters.to_openai_kwargs()

        # Add provider-specific parameters
        provider_params = cast(OpenAICompletionParams, request.provider_params)
        openai_kwargs.update(
            {k: v for k, v in provider_params.items() if v is not None}
        )

        try:
            # Request timeout from parameters or default
            timeout = openai_kwargs.pop("timeout", 30)

            # Make the API call
            response = self.client.completions.create(
                model=self.model_info.name,
                timeout=timeout,
                **openai_kwargs,
            )

            # Extract completion text
            text = response.choices[0].text.strip()

            # Calculate usage statistics
            # For simplicity, we assume a usage calculator is implemented elsewhere
            usage_stats = (
                None  # self.usage_calculator.calculate(response, self.model_info)
            )

            return CompletionResponse(
                text=text,
                raw_output=response,
                usage=usage_stats,
            )

        except Exception as exc:
            logger.exception("Unexpected error in OpenAIExtendedModel.complete()")
            raise ProviderAPIError(str(exc)) from exc

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generating embeddings for the input text(s).

        Implements embedding capabilities using the OpenAI embeddings API.

        Args:
            request: Embedding request with input text(s).

        Returns:
            Embedding response with vector representations.

        Raises:
            InvalidPromptError: If input is empty.
            ProviderAPIError: For unexpected errors during API calls.
        """
        # Use the provided model or default to the model in model_info
        model_name = request.model or self.model_info.name

        input_text = request.input
        if not input_text:
            raise InvalidPromptError("Input text for embeddings cannot be empty.")

        logger.info(
            "OpenAI embeddings invoked",
            extra={
                "provider": self.PROVIDER_NAME,
                "model_name": model_name,
                "input_type": "batch" if isinstance(input_text, list) else "single",
            },
        )

        try:
            # Make the API call
            response = self.client.embeddings.create(
                model=model_name,
                input=input_text,
                timeout=30,
            )

            # Extract embeddings
            if isinstance(input_text, list):
                # For batch processing
                embeddings = [item.embedding for item in response.data]
            else:
                # For single text input
                embeddings = response.data[0].embedding

            # Get dimensions from the first embedding
            if isinstance(embeddings, list) and isinstance(embeddings[0], list):
                dimensions = len(embeddings[0])
            else:
                dimensions = len(embeddings)

            # Calculate usage statistics (implementation would depend on your system)
            usage_stats = (
                None  # self.usage_calculator.calculate(response, self.model_info)
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model_name,
                dimensions=dimensions,
                raw_output=response,
                usage=usage_stats,
            )

        except Exception as exc:
            logger.exception("Unexpected error in OpenAIExtendedModel.embed()")
            raise ProviderAPIError(str(exc)) from exc


# -----------------------------------------------------------------------------
# PART 5: Model Registry Extension Blueprint
# -----------------------------------------------------------------------------


class EnhancedModelRegistry(Generic[ModelT]):
    """Enhanced model registry with multi-capability support.

    This extension demonstrates how to modify the existing ModelRegistry to
    support multiple model capabilities while maintaining backward compatibility.

    Attributes:
        _lock: Thread lock ensuring thread-safe operations.
        _models: Mapping from model IDs to model instances.
        _model_infos: Mapping from model IDs to their metadata.
        _completion_models: Mapping from model IDs to completion-capable models.
        _embedding_models: Mapping from model IDs to embedding-capable models.
        _logger: Logger instance specific to this registry.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initializing a new instance of EnhancedModelRegistry.

        Args:
            logger: Optional logger to use. If None, a default logger is created.
        """
        # The following would match the existing ModelRegistry implementation
        # with added capability-specific registries
        self._lock = threading.Lock()
        self._model_infos: Dict[str, ModelInfo] = {}
        self._models: Dict[str, ModelT] = {}
        self._completion_models: Dict[str, TextCompletionCapable] = {}
        self._embedding_models: Dict[str, EmbeddingCapable] = {}
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def _get_capabilities(self, model: Any) -> Dict[str, bool]:
        """Determining the capabilities of a model instance.

        Args:
            model: A model instance to check for capabilities.

        Returns:
            Dictionary mapping capability names to boolean support flags.
        """
        # Preferred: check for CAPABILITIES class attribute
        if hasattr(model, "CAPABILITIES") and isinstance(model.CAPABILITIES, dict):
            return model.CAPABILITIES

        # Fallback: check for protocol implementation
        return {
            "chat": isinstance(model, BaseProviderModel),
            "completion": isinstance(model, TextCompletionCapable),
            "embedding": isinstance(model, EmbeddingCapable),
        }

    def register_model(self, model_info: ModelInfo) -> None:
        """Registering a new model using its metadata.

        This method would extend the existing register_model implementation
        to detect and store model capabilities.

        Args:
            model_info: The configuration and metadata for the model.

        Raises:
            ValueError: If a model with the same ID is already registered.
        """
        # This would contain the standard ModelRegistry implementation
        # with added capability detection
        pass

    def get_completion_model(self, model_id: str) -> TextCompletionCapable:
        """Getting a text completion capable model by ID.

        Args:
            model_id: Unique identifier of the model.

        Returns:
            A model instance implementing the TextCompletionCapable protocol.

        Raises:
            ValueError: If model_id is empty.
            ModelNotFoundError: If the model is not registered.
            TypeError: If the model does not support text completion.
        """
        # This would lazily instantiate the model if needed
        # and verify it supports text completion capabilities
        pass

    def get_embedding_model(self, model_id: str) -> EmbeddingCapable:
        """Getting an embedding capable model by ID.

        Args:
            model_id: Unique identifier of the model.

        Returns:
            A model instance implementing the EmbeddingCapable protocol.

        Raises:
            ValueError: If model_id is empty.
            ModelNotFoundError: If the model is not registered.
            TypeError: If the model does not support embedding capabilities.
        """
        # This would lazily instantiate the model if needed
        # and verify it supports embedding capabilities
        pass


# -----------------------------------------------------------------------------
# PART 6: Usage Examples
# -----------------------------------------------------------------------------


def example_text_completion_usage() -> None:
    """Demonstrating text completion usage.

    Shows how the extended API would be used to interact with text completion models.
    """
    # Example assumes you have providers registered and configured
    # Typically you would access these through a registry service
    from ember.api import models

    # Direct method call (requires the provider to implement TextCompletionCapable)
    response = models.openai_extended.complete_text(
        "The best way to learn programming is to",
        max_tokens=50,
        temperature=0.7,
        stop_sequences=["\n\n"],
    )
    print(f"Completion: {response.text}")

    # Using namespace approach (if API layer is extended for completion)
    # This would require extending the models API to support completion
    # response = models.completion.openai_gpt35("The best way to learn programming is to")
    # print(f"Completion: {response.text}")


def example_embedding_usage() -> None:
    """Demonstrating embeddings usage.

    Shows how the extended API would be used to interact with embedding models.
    """
    # Example assumes you have providers registered and configured
    from ember.api import models

    # Single text embedding
    response = models.openai_extended.embed_text(
        "Embeddings capture semantic meaning of text"
    )
    print(f"Embedding dimensions: {response.dimensions}")
    print(f"First 5 values: {response.embeddings[:5]}")

    # Batch text embedding
    batch_response = models.openai_extended.embed_text(
        ["Embeddings are useful", "For semantic search", "And clustering tasks"]
    )
    print(f"Batch embeddings count: {len(batch_response.embeddings)}")

    # Calculate similarity between two embeddings
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculating cosine similarity between two vectors."""
        a_array = np.array(a)
        b_array = np.array(b)
        return np.dot(a_array, b_array) / (
            np.linalg.norm(a_array) * np.linalg.norm(b_array)
        )

    text1 = "Machine learning is fascinating"
    text2 = "AI is an exciting field"

    embed_response1 = models.openai_extended.embed_text(text1)
    embed_response2 = models.openai_extended.embed_text(text2)

    similarity = cosine_similarity(
        embed_response1.embeddings, embed_response2.embeddings  # type: ignore
    )
    print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")


if __name__ == "__main__":
    print(
        "This module provides examples and guidelines for extending the model registry."
    )
    print("It demonstrates support for text completion and embedding models.")

    # These would be run if the module was executed directly
    # example_text_completion_usage()
    # example_embedding_usage()
