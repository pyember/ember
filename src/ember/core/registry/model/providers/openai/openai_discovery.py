"""
OpenAI model discovery provider.

This module implements model discovery using the latest OpenAI Python SDK.
It creates a client via the OpenAI class, retrieves available models using
client.models.list(), filters the results, and standardizes them for the Ember model registry.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from openai import APIError, OpenAI

from ember.core.exceptions import ModelDiscoveryError
from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider

# Module-level logger.
logger = logging.getLogger(__name__)
# Set default log level to WARNING to reduce verbosity
logger.setLevel(logging.WARNING)


class OpenAIDiscovery(BaseDiscoveryProvider):
    """Discovery provider for OpenAI models.

    Retrieves models from the OpenAI API using the latest OpenAI SDK patterns.
    """

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        """
        Initialize the OpenAIDiscovery instance.

        The API key is provided either as an argument or via the OPENAI_API_KEY environment variable.
        Optionally, a custom base URL (for example for Azure endpoints) may be specified.

        Args:
            api_key (Optional[str]): The OpenAI API key for authentication.
            base_url (Optional[str]): An optional custom base URL for the OpenAI API.

        Raises:
            ModelDiscoveryError: If the API key is not set.
        """
        self._api_key: Optional[str] = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ModelDiscoveryError(
                "OpenAI API key is not set in config or environment"
            )
        self._base_url: Optional[str] = base_url

        client_kwargs: Dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url

        # Instantiate the OpenAI client using the latest SDK.
        self.client: OpenAI = OpenAI(**client_kwargs)

        # List of model prefixes used for filtering relevant models.
        self._model_filter_prefixes: List[str] = [
            "gpt-4",
            "gpt-3.5",
            "text-embedding",
            "dall-e",
        ]

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch and standardize OpenAI models from the API.

        Retrieves models via the OpenAI client, filters them to include only relevant models,
        and converts the data into a standardized mapping suitable for the Ember registry.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping from standardized model IDs to model details.

        Raises:
            ModelDiscoveryError: If API access fails and no models can be retrieved.
        """
        try:
            models_page: List[Any] = list(self.client.models.list())
            model_list: List[Dict[str, Any]] = [
                {"id": model.id, "object": model.object} for model in models_page
            ]
            logger.debug("Fetched %d models from OpenAI API", len(model_list))

            filtered_models: List[Dict[str, Any]] = self._filter_models(model_list)
            logger.debug("Filtered to %d relevant models", len(filtered_models))

            standardized_models: Dict[str, Dict[str, Any]] = {}
            for raw_model in filtered_models:
                raw_model_id: str = raw_model.get("id", "")
                standardized_id: str = self._generate_model_id(raw_model_id)
                standardized_models[standardized_id] = self._build_model_entry(
                    model_id=standardized_id, model_data=raw_model
                )

            # Return discovered models, even if empty
            if not standardized_models:
                logger.warning(
                    "No OpenAI models found after filtering - API discovery required"
                )

            return standardized_models

        except APIError as api_err:
            logger.error("OpenAI API error: %s", api_err)
            logger.warning("No fallback models provided - API discovery required")
            return {}
        except Exception as unexpected_err:
            logger.exception(
                "Unexpected error fetching OpenAI models: %s", unexpected_err
            )
            logger.warning("No fallback models provided - API discovery required")
            return {}

    def _filter_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter models to include only those with recognized prefixes.

        Args:
            models (List[Dict[str, Any]]): List of model dictionaries.

        Returns:
            List[Dict[str, Any]]: Filtered list of model dictionaries.
        """
        return [
            model
            for model in models
            if any(
                model.get("id", "").startswith(prefix)
                for prefix in self._model_filter_prefixes
            )
        ]

    def _generate_model_id(self, raw_model_id: str) -> str:
        """
        Generate a standardized model ID with the 'openai:' prefix.

        Args:
            raw_model_id (str): The raw model identifier from the API.

        Returns:
            str: The standardized model identifier.
        """
        return f"openai:{raw_model_id}"

    def _build_model_entry(
        self, model_id: str, model_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construct a standardized model entry.

        Args:
            model_id (str): The standardized model identifier.
            model_data (Dict[str, Any]): The raw model data from the OpenAI API.

        Returns:
            Dict[str, Any]: A dictionary containing the standardized model details.
        """
        return {
            "id": model_id,
            "name": model_data.get("id", ""),
            "api_data": model_data,
        }

    # Fallback methods removed in favor of direct API discovery
