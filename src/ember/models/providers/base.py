"""Contracts that model providers must implement to plug into Ember.

Only the minimal interface lives here so providers can tailor their own
feature surface. Consumers rely on this module for type hints and shared
documentation.

Examples:
    >>> from ember.models.providers.base import BaseProvider
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generator, Mapping, Optional, Sequence, Tuple

from ember._internal.exceptions import ProviderAPIError
from ember.core.credentials import CredentialNotFoundError

if TYPE_CHECKING:
    # Imported only for type checking to avoid circular imports at runtime.
    from ember.models.schemas import ChatResponse, EmbeddingResponse


@dataclass(frozen=True)
class ProviderCapacity:
    """Static capacity information for a provider shard."""

    rpm: Optional[int] = None
    burst: Optional[int] = None
    max_concurrency: Optional[int] = None
    regions: Tuple[str, ...] = field(default_factory=tuple)


class BaseProvider(ABC):
    """Abstract contract that all Ember model providers implement.

    Providers must supply completion logic and their own credential lookup so
    the registry can treat every backend uniformly.

    Attributes:
        api_key: Credential used for authenticated providers.
    """

    # Providers can override to opt-out of credential requirements (e.g., local runtimes)
    requires_api_key: bool = True
    # Providers that implement the OpenAI Responses API set this to True.
    supports_responses_api: bool = False
    capacity: ProviderCapacity = ProviderCapacity()

    def __init__(self, api_key: str | None = None):
        """Initialize the provider and resolve credentials.

        Args:
            api_key: Optional credential overriding configured credential lookup.

        Raises:
            ValueError: If credentials are required but unavailable.
        """
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError(f"api_key must be a string, got {type(api_key).__name__}")

        cleaned_key = api_key.strip() if isinstance(api_key, str) else ""

        # Allow keyless providers to initialize without credentials
        if self.requires_api_key:
            message = (
                f"API key required for {self.__class__.__name__}. "
                "Set providers.<provider>.api_key in ~/.ember/config.yaml (use `ember configure`) "
                "or pass api_key=... explicitly."
            )
            try:
                resolved_key = cleaned_key or self._get_api_key_from_env()
            except CredentialNotFoundError as exc:
                raise ValueError(message) from exc

            resolved_key = resolved_key.strip()

            self.api_key = resolved_key
            if not self.api_key:
                raise ValueError(message)
        else:
            self.api_key = cleaned_key  # may be empty by design

    @abstractmethod
    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        """Return a ChatResponse produced by the underlying provider.

        Args:
            prompt: Prompt text in provider-native format.
            model: Provider-specific model identifier.
            **kwargs: Provider-defined options (temperature, tools, etc.).

        Returns:
            ChatResponse: Normalized response payload.

        Raises:
            ProviderAPIError: If the provider reports an error.
        """
        pass

    def stream_complete(
        self, prompt: str, model: str, **kwargs: Any
    ) -> Generator[str, None, ChatResponse]:
        """Yield streamed text chunks and return the final ChatResponse.

        Providers that support streaming should override this method. The default
        implementation raises a ProviderAPIError to surface a clear failure to
        callers rather than silently degrading to non-streaming behavior.

        Args:
            prompt: Prompt text in provider-native format.
            model: Provider-specific model identifier.
            **kwargs: Provider-defined options (temperature, tools, etc.).

        Raises:
            ProviderAPIError: Always, indicating that streaming is unsupported.
        """
        raise ProviderAPIError(
            "Streaming is not implemented for this provider",
            context={"model": model, "provider": type(self).__name__},
        )

    def complete_responses_payload(
        self,
        payload: Mapping[str, object],
        **kwargs: Any,
    ) -> ChatResponse:
        """Handle a pre-built Responses API payload.

        Providers that support the Responses API must override this method.
        """

        raise ProviderAPIError(
            "Responses API is not implemented for this provider",
            context={"provider": type(self).__name__},
        )

    def stream_responses_payload(
        self,
        payload: Mapping[str, object],
        **kwargs: Any,
    ) -> Generator[str, None, ChatResponse]:
        """Stream a Responses API payload."""

        raise ProviderAPIError(
            "Responses API streaming is not implemented for this provider",
            context={"provider": type(self).__name__},
        )

    def embed(self, inputs: Sequence[str], model: str, **kwargs: Any) -> EmbeddingResponse:
        """Return embeddings for the supplied inputs.

        Providers should override this when they support embedding endpoints.
        The default implementation raises to surface the unsupported capability.
        """
        raise ProviderAPIError(
            "Embeddings are not implemented for this provider",
            context={"model": model, "provider": type(self).__name__},
        )

    @abstractmethod
    def _get_api_key_from_env(self) -> str:
        """Return a credential looked up from the active Ember configuration.

        Returns:
            str: API key.
        """
        pass

    def validate_model(self, model: str) -> bool:
        """Return True when ``model`` is supported by this provider.

        Subclasses can override to short-circuit invalid requests instead of
        relying on vendor errors.

        Args:
            model: Model identifier supplied by the caller.

        Returns:
            bool: ``True`` when the provider supports ``model``.
        """
        return True

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Return provider-specific metadata for ``model``.

        Args:
            model: Model identifier in provider syntax.

        Returns:
            Dict[str, Any]: Metadata fields describing ``model``.
        """
        return {
            "model": model,
            "provider": self.__class__.__name__,
        }

    @classmethod
    def get_capacity(cls) -> ProviderCapacity:
        """Return declared capacity details for the provider."""

        return cls.capacity
