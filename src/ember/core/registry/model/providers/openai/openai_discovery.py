"""OpenAI model discovery provider.

This module implements the discovery of OpenAI models from the API,
supporting both modern (v1+) and legacy versions of the OpenAI SDK.
It transforms raw API responses into a standardized format for the Ember model registry.
"""

import os
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..base_discovery import BaseDiscoveryProvider, ModelDiscoveryError

# Module-level logger.
logger = logging.getLogger(__name__)


@runtime_checkable
class OpenAIClientProtocol(Protocol):
    """Protocol defining the interface of an OpenAI client.

    This abstraction enables support for multiple OpenAI SDK versions without
    binding to a specific implementation.
    """

    @abstractmethod
    def list_models(self) -> Any:
        """Retrieve the list of available models from the OpenAI API.

        Returns:
            Any: The raw API response containing model information.
        """
        ...


class OpenAIDiscovery(BaseDiscoveryProvider):
    """Discovery provider for OpenAI models.

    Retrieves models from the OpenAI API using either the modern (v1+) or legacy SDK,
    and converts them into a standardized format for the Ember model registry.
    """

    def __init__(self) -> None:
        """Initialize the OpenAIDiscovery instance."""
        self._api_key: Optional[str] = None
        self._base_url: Optional[str] = None
        self._client: Optional[OpenAIClientProtocol] = None
        self._model_filter_prefixes: List[str] = [
            "gpt-4", "gpt-3.5", "text-embedding", "dall-e"
        ]

    def configure(self, api_key: str, base_url: Optional[str] = None) -> None:
        """Configure the discovery provider with API credentials.

        Args:
            api_key (str): The OpenAI API key for authentication.
            base_url (Optional[str]): An optional custom base URL for the OpenAI API.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # Reset client to force re-creation with new credentials.

    def _get_client(self) -> OpenAIClientProtocol:
        """Retrieve or instantiate an OpenAI client.

        Attempts to create a client using the modern SDK first. If the modern SDK is not
        available (ImportError), falls back to the legacy SDK.

        Returns:
            OpenAIClientProtocol: An instance capable of listing models.

        Raises:
            ModelDiscoveryError: If API key is missing or initialization fails.
        """
        if self._client is not None:
            return self._client

        if self._api_key is None:
            self._api_key = os.environ.get("OPENAI_API_KEY")
            if not self._api_key:
                raise ModelDiscoveryError("OpenAI API key is not set")

        try:
            self._client = self._create_modern_client()
            return self._client
        except ImportError as imp_err:
            logger.debug("Modern OpenAI SDK not available: %s", imp_err)
        except ModelDiscoveryError as mod_err:
            logger.debug("Modern OpenAI SDK test call failed: %s", mod_err)
            raise mod_err

        try:
            self._client = self._create_legacy_client()
            return self._client
        except ImportError as imp_err:
            logger.error("No compatible OpenAI SDK found: %s", imp_err)
            raise ModelDiscoveryError("No compatible OpenAI SDK found")
        except Exception as unexpected_err:
            logger.exception("Unexpected error initializing OpenAI client: %s", unexpected_err)
            raise ModelDiscoveryError(f"OpenAI client initialization failed: {unexpected_err}")

    def _create_modern_client(self) -> OpenAIClientProtocol:
        """Create an OpenAI client using the modern SDK (v1+).

        Returns:
            OpenAIClientProtocol: An adapter wrapping the modern OpenAI client.

        Raises:
            ModelDiscoveryError: If the API test call fails.
            ImportError: If the modern OpenAI SDK is not installed.
        """
        from openai import OpenAI, APIError  # May raise ImportError.
        client_kwargs: Dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url

        client_instance: Any = OpenAI(**client_kwargs)

        # Skip API test calls during initialization to avoid failures
        logger.debug("Using OpenAI v1+ SDK client (no connectivity test)")
        # We'll let the actual fetch_models call handle any API errors

        class _OpenAIClientV1Adapter(OpenAIClientProtocol):
            """Adapter for the modern OpenAI SDK client."""

            def __init__(self, client: Any) -> None:
                """Initialize the adapter.

                Args:
                    client (Any): The raw modern OpenAI client instance.
                """
                self._client: Any = client

            def list_models(self) -> Any:
                """List models using the modern SDK client.

                Returns:
                    Any: The API response containing model information.
                """
                return self._client.models.list()

        return _OpenAIClientV1Adapter(client_instance)

    def _create_legacy_client(self) -> OpenAIClientProtocol:
        """Create an OpenAI client using the legacy SDK.

        Returns:
            OpenAIClientProtocol: An adapter wrapping the legacy OpenAI client.

        Raises:
            ModelDiscoveryError: If the API test call fails.
            ImportError: If the legacy OpenAI SDK is not installed.
        """
        import openai  # May raise ImportError.
        openai.api_key = self._api_key
        if self._base_url is not None:
            openai.api_base = self._base_url

        class _OpenAILegacyClientAdapter(OpenAIClientProtocol):
            """Adapter for the legacy OpenAI SDK client."""

            def list_models(self) -> Any:
                """List models using the legacy SDK client.

                Returns:
                    Any: The API response containing model information.
                """
                return openai.Model.list()

        # Skip API test calls during initialization to avoid failures
        logger.debug("Using OpenAI legacy SDK client (no connectivity test)")
        # We'll let the actual fetch_models call handle any API errors

        return _OpenAILegacyClientAdapter()

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch and standardize OpenAI models from the API.

        Retrieves models using the OpenAI client, filters relevant models,
        and converts them into a standardized mapping suitable for the Ember registry.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping from standardized model IDs to model details.

        Raises:
            ModelDiscoveryError: If the OpenAI API access fails.
        """
        try:
            client: OpenAIClientProtocol = self._get_client()
            response: Any = client.list_models()

            model_list: List[Dict[str, Any]] = []
            if hasattr(response, "data") and isinstance(response.data, list):
                # Modern SDK response format.
                model_list = [
                    {"id": model.id, "object": model.object}
                    for model in response.data
                ]
            elif isinstance(response, dict) and "data" in response:
                # Legacy SDK response format.
                model_list = response.get("data", [])
            else:
                logger.warning("Unexpected OpenAI API response format: %s", response)
                return {}

            logger.debug("Fetched %d models from OpenAI API", len(model_list))
            filtered_models: List[Dict[str, Any]] = self._filter_models(models=model_list)
            logger.debug("Filtered to %d relevant models", len(filtered_models))

            standardized_models: Dict[str, Dict[str, Any]] = {}
            for raw_model in filtered_models:
                raw_model_id: str = raw_model.get("id", "")
                standardized_id: str = self._generate_model_id(raw_model_id=raw_model_id)
                standardized_models[standardized_id] = self._build_model_entry(
                    model_id=standardized_id, model_data=raw_model
                )

            if not standardized_models:
                logger.info("No OpenAI models found after filtering; adding fallback models")
                self._add_fallback_models(models_dict=standardized_models)

            return standardized_models

        except ModelDiscoveryError as disc_err:
            logger.error("OpenAI model discovery error: %s", disc_err)
            return self._get_fallback_models()
        except Exception as unexpected_err:
            logger.exception("Unexpected error fetching OpenAI models: %s", unexpected_err)
            return self._get_fallback_models()

    def _filter_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter models to include only those with recognized prefixes.

        Args:
            models (List[Dict[str, Any]]): List of model dictionaries.

        Returns:
            List[Dict[str, Any]]: Filtered list of model dictionaries.
        """
        return [
            model for model in models
            if any(model.get("id", "").startswith(prefix) for prefix in self._model_filter_prefixes)
        ]

    def _generate_model_id(self, raw_model_id: str) -> str:
        """Generate a standardized model ID with the 'openai:' prefix.

        Args:
            raw_model_id (str): The raw model identifier from the API.

        Returns:
            str: The standardized model identifier.
        """
        return f"openai:{raw_model_id}"

    def _build_model_entry(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a standardized model entry.

        Args:
            model_id (str): The standardized model identifier.
            model_data (Dict[str, Any]): The raw model data from the OpenAI API.

        Returns:
            Dict[str, Any]: A dictionary containing the standardized model details.
        """
        return {
            "model_id": model_id,
            "model_name": model_data.get("id", ""),
            "api_data": model_data,
        }

    def _add_fallback_models(self, models_dict: Dict[str, Dict[str, Any]]) -> None:
        """Add fallback models to the provided dictionary.

        Args:
            models_dict (Dict[str, Dict[str, Any]]): The dictionary to populate with fallback models.
        """
        fallback_models: Dict[str, Dict[str, Any]] = self._get_fallback_models()
        for model_id, model_data in fallback_models.items():
            if model_id not in models_dict:
                models_dict[model_id] = model_data

    def _get_fallback_models(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve fallback models for cases when API discovery fails.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing fallback model data.
        """
        return {
            "openai:gpt-4o": {
                "model_id": "openai:gpt-4o",
                "model_name": "gpt-4o",
                "api_data": {"object": "model"}
            },
            "openai:gpt-4o-mini": {
                "model_id": "openai:gpt-4o-mini",
                "model_name": "gpt-4o-mini",
                "api_data": {"object": "model"}
            },
            "openai:gpt-3.5-turbo": {
                "model_id": "openai:gpt-3.5-turbo",
                "model_name": "gpt-3.5-turbo",
                "api_data": {"object": "model"}
            }
        }
