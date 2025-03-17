"""
Anthropic model discovery provider.

This module implements model discovery using the latest Anthropic Python SDK.
It creates a client via the Anthropic class, retrieves available models using
client.models.list(), and standardizes them for the Ember model registry.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from anthropic import (
    Anthropic,
    APIError,
    RateLimitError,
    APIStatusError,
    APIConnectionError,
)
from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError,
)

# Module-level logger.
logger = logging.getLogger(__name__)


class AnthropicDiscovery(BaseDiscoveryProvider):
    """
    Discovery provider for Anthropic models using the latest SDK.

    Retrieves available models from the Anthropic API via client.models.list()
    and standardizes them for the Ember model registry.
    """

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        """
        Initialize the AnthropicDiscovery instance.

        The API key is provided either as an argument or via the ANTHROPIC_API_KEY
        environment variable. An optional base URL can be specified (for custom endpoints).

        Args:
            api_key (Optional[str]): The Anthropic API key for authentication.
            base_url (Optional[str]): An optional custom base URL for the Anthropic API.

        Raises:
            ModelDiscoveryError: If the API key is not set.
        """
        self._api_key: Optional[str] = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ModelDiscoveryError(
                "Anthropic API key is not set in config or environment"
            )
        self._base_url: Optional[str] = base_url
        client_kwargs: Dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url

        # Instantiate the Anthropic client using the latest SDK.
        self.client: Anthropic = Anthropic(**client_kwargs)

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch Anthropic model metadata.

        Retrieves models via the Anthropic client's models.list() method,
        standardizes them, and returns a mapping from canonical model IDs to model details.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping standardized model IDs to their metadata.
        """
        try:
            models_page = self.client.models.list()
            models_data: List[
                Any
            ] = models_page.data  # models_page.data is a list of model objects.
            standardized_models: Dict[str, Dict[str, Any]] = {}
            for model in models_data:
                raw_model_id: str = model.id
                # Extract base version without date for standardized model ID
                base_model_id = self._extract_base_model_id(raw_model_id)
                canonical_id: str = self._generate_model_id(base_model_id)
                standardized_models[canonical_id] = self._build_model_entry(
                    model_id=canonical_id,
                    model_data={
                        "id": raw_model_id,
                        "object": getattr(model, "object", "model"),
                    },
                )

            if not standardized_models:
                logger.info("No Anthropic models found; adding fallback models")
                self._add_fallback_models(standardized_models)
            return standardized_models

        except APIError as api_err:
            logger.error("Anthropic API error fetching models: %s", api_err)
            return self._get_fallback_models()
        except Exception as unexpected_err:
            logger.exception(
                "Unexpected error fetching Anthropic models: %s", unexpected_err
            )
            return self._get_fallback_models()

    def _generate_model_id(self, raw_model_id: str) -> str:
        """
        Generate a standardized model ID with the 'anthropic:' prefix.

        Args:
            raw_model_id (str): The raw model identifier from the API.

        Returns:
            str: The standardized model identifier.
        """
        return f"anthropic:{raw_model_id}"

    def _build_model_entry(
        self, *, model_id: str, model_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construct a standardized model entry.

        Args:
            model_id (str): The standardized model identifier.
            model_data (Dict[str, Any]): The raw model data.

        Returns:
            Dict[str, Any]: A dictionary containing the standardized model details.
        """
        return {
            "model_id": model_id,
            "model_name": model_data.get("id", model_id.split(":")[-1]),
            "api_data": model_data,
        }

    def _add_fallback_models(self, models_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Add fallback models to the provided dictionary if they are missing.

        Args:
            models_dict (Dict[str, Dict[str, Any]]): The dictionary to which fallback models are added.
        """
        fallback_models: Dict[str, Dict[str, Any]] = self._get_fallback_models()
        for model_id, model_data in fallback_models.items():
            if model_id not in models_dict:
                models_dict[model_id] = model_data

    def _get_fallback_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve fallback models when API discovery fails.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing fallback model data.
        """
        return {
            "anthropic:claude-3": {
                "model_id": "anthropic:claude-3",
                "model_name": "claude-3",
                "api_data": {"object": "model"},
            },
            "anthropic:claude-3.5": {
                "model_id": "anthropic:claude-3.5",
                "model_name": "claude-3.5",
                "api_data": {"object": "model"},
            },
        }
