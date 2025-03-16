"""Anthropic model discovery provider.

This module provides the implementation for discovering available models from
the Anthropic API and transforming the data into a standardized format for the
Ember framework.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Protocol, cast, runtime_checkable

from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError,
)
from ember.core.app_context import get_app_context

# Logger for this module.
logger = logging.getLogger(__name__)

# Type aliases for improved clarity.
ModelDict = Dict[str, Any]
ModelList = List[ModelDict]
ModelListResponse = Dict[str, ModelList]


@runtime_checkable
class AnthropicClientProtocol(Protocol):
    """Protocol defining the interface expected from an Anthropic client.

    This protocol allows support for different client implementations without
    imposing direct dependencies on a specific implementation.
    """

    def list_models(self) -> ModelListResponse:
        """List available models from the Anthropic API.

        Returns:
            ModelListResponse: The raw API response containing model information.
        """
        ...


class _AnthropicClientAdapter:
    """Adapter for the Anthropic client to simulate model listing functionality.

    This adapter wraps the actual Anthropic client and provides a standardized
    interface expected by the discovery provider.
    """

    def __init__(self, *, api_key: str, base_url: Optional[str] = None) -> None:
        """Initialize the Anthropic client adapter.

        Args:
            api_key (str): The API key for authentication.
            base_url (Optional[str]): Optional custom base URL for the Anthropic API.
        """
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        try:
            import anthropic
        except ImportError as import_err:
            logger.exception("Failed to import the Anthropic SDK: %s", import_err)
            raise
        self._client = anthropic.Anthropic(**kwargs)

    def list_models(self) -> ModelListResponse:
        """Return a simulated model listing response.

        Since the Anthropic API does not support model listing directly,
        returns a static list of known models.

        Returns:
            ModelListResponse: Dictionary with a 'data' key mapping to a list of model details.
        """
        return {"data": self._get_known_models()}

    def _get_known_models(self) -> ModelList:
        """Retrieve a list of known Anthropic models.

        Returns:
            ModelList: A list of dictionaries, each representing a model.
        """
        return [
            {"id": "claude-3-opus-20240229", "object": "model"},
            {"id": "claude-3-sonnet-20240229", "object": "model"},
            {"id": "claude-3-haiku-20240307", "object": "model"},
            {"id": "claude-3.5-sonnet-20240620", "object": "model"},
        ]


class AnthropicDiscovery(BaseDiscoveryProvider):
    """Discovery provider for Anthropic models.

    Retrieves available models from the Anthropic API (if available) or uses
    pre-defined model information when the API doesn't support model discovery.
    """

    def __init__(self) -> None:
        """Initialize the Anthropic discovery provider."""
        self._api_key: Optional[str] = None
        self._base_url: Optional[str] = None
        self._client: Optional[AnthropicClientProtocol] = None

    def configure(self, *, api_key: str, base_url: Optional[str] = None) -> None:
        """Configure the discovery provider with API credentials.

        Args:
            api_key (str): The Anthropic API key for authentication.
            base_url (Optional[str]): Optional custom base URL for the Anthropic API.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # Reset client to force re-creation with new credentials

    def _get_client(self) -> Optional[AnthropicClientProtocol]:
        """Retrieve or create an Anthropic client instance if possible.

        Note:
            The Anthropic API currently does not support listing models,
            so this method may return None if no compatible client can be created.

        Returns:
            Optional[AnthropicClientProtocol]: A client instance or None if not available.
        """
        if self._client is not None:
            return self._client

        # Try to get API key from configuration if not provided
        if not self._api_key:
            try:
                # Get from centralized config
                app_context = get_app_context()
                provider_config = app_context.config_manager.get_config().get_provider(
                    "anthropic"
                )
                if provider_config and provider_config.api_keys.get("default"):
                    # Access the key attribute from the API key object
                    default_key = provider_config.api_keys["default"]
                    if hasattr(default_key, "key"):
                        self._api_key = default_key.key
                    # Also get base_url if not already specified
                    if not self._base_url and hasattr(provider_config, "base_url"):
                        self._base_url = provider_config.base_url
            except Exception as config_error:
                logger.debug(f"Could not get API key from config: {config_error}")

            # Fallback to environment variable
            if not self._api_key:
                self._api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not self._api_key:
                logger.warning(
                    "No Anthropic API key found in config or environment variables"
                )
                return None

        try:
            self._client = _AnthropicClientAdapter(
                api_key=self._api_key, base_url=self._base_url
            )
            logger.debug("Created Anthropic client adapter (using static model list)")
            return self._client
        except ImportError:
            logger.debug("Anthropic SDK not available")
            return None
        except Exception as unexpected_error:
            logger.exception(
                "Unexpected error initializing Anthropic client: %s", unexpected_error
            )
            return None

    def fetch_models(self) -> Dict[str, ModelDict]:
        """Fetch Anthropic model metadata.

        Returns:
            Dict[str, ModelDict]: A dictionary mapping model IDs to their metadata.
            Each metadata dictionary contains keys such as 'model_id', 'model_name',
            and 'api_data'.
        """
        try:
            client: Optional[AnthropicClientProtocol] = self._get_client()
            models_data: List[ModelDict] = []

            if client is not None:
                try:
                    response: ModelListResponse = client.list_models()
                    if isinstance(response, dict) and "data" in response:
                        models_data = cast(List[ModelDict], response["data"])
                    else:
                        logger.warning(
                            "Unexpected response format from Anthropic client: %s",
                            response,
                        )
                except Exception as client_error:
                    logger.error(
                        "Error fetching models from Anthropic client: %s", client_error
                    )

            standardized_models: Dict[str, ModelDict] = {}
            for model in models_data:
                model_id_raw: str = model.get("id", "")
                if model_id_raw:
                    canonical_id: str = self._generate_model_id(
                        raw_model_id=model_id_raw
                    )
                    standardized_models[canonical_id] = self._build_model_entry(
                        model_id=canonical_id, model_data=model
                    )

            self._add_fallback_models(models_dict=standardized_models)
            logger.debug(
                "Returning %d Anthropic models: %s",
                len(standardized_models),
                list(standardized_models.keys()),
            )
            return standardized_models

        except Exception as error:
            logger.exception("Failed to fetch models from Anthropic: %s", error)
            return self._get_fallback_models()

    def _generate_model_id(self, *, raw_model_id: str) -> str:
        """Generate a standardized model ID with the 'anthropic:' prefix.

        This method extracts the short model name from its full ID (e.g.,
        "claude-3-opus-20240229" becomes "claude-3-opus").

        Args:
            raw_model_id (str): The raw model identifier from the API.

        Returns:
            str: The standardized model identifier.
        """
        base_name: str = raw_model_id.split("-2024")[0]
        return f"anthropic:{base_name}"

    def _build_model_entry(self, *, model_id: str, model_data: ModelDict) -> ModelDict:
        """Construct a standardized model entry.

        Args:
            model_id (str): The standardized model identifier.
            model_data (ModelDict): The raw model data.

        Returns:
            ModelDict: A dictionary containing model details.
        """
        return {
            "model_id": model_id,
            "model_name": model_data.get("id", model_id.split(":")[-1]),
            "api_data": model_data,
        }

    def _add_fallback_models(self, *, models_dict: Dict[str, ModelDict]) -> None:
        """Add fallback models to the provided dictionary if they are missing.

        Args:
            models_dict (Dict[str, ModelDict]): The dictionary to which fallback models are added.
        """
        fallback_models: Dict[str, ModelDict] = self._get_fallback_models()
        for model_id, model_data in fallback_models.items():
            if model_id not in models_dict:
                models_dict[model_id] = model_data

    def _get_fallback_models(self) -> Dict[str, ModelDict]:
        """Retrieve fallback models when API discovery fails.

        Returns:
            Dict[str, ModelDict]: A dictionary containing fallback model data.
        """
        return {
            "anthropic:claude-3-opus": {
                "model_id": "anthropic:claude-3-opus",
                "model_name": "claude-3-opus",
                "api_data": {"object": "model"},
            },
            "anthropic:claude-3-sonnet": {
                "model_id": "anthropic:claude-3-sonnet",
                "model_name": "claude-3-sonnet",
                "api_data": {"object": "model"},
            },
            "anthropic:claude-3-haiku": {
                "model_id": "anthropic:claude-3-haiku",
                "model_name": "claude-3-haiku",
                "api_data": {"object": "model"},
            },
            "anthropic:claude-3.5-sonnet": {
                "model_id": "anthropic:claude-3.5-sonnet",
                "model_name": "claude-3.5-sonnet",
                "api_data": {"object": "model"},
            },
        }
