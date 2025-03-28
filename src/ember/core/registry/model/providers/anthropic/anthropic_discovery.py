"""
Anthropic model discovery provider.

This module implements model discovery using direct HTTP requests to the Anthropic API
endpoints. It retrieves available models using the /v1/models endpoint and
standardizes them for the Ember model registry.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from anthropic import Anthropic

from ember.core.exceptions import ModelDiscoveryError
from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider

# Module-level logger.
logger = logging.getLogger(__name__)
# Set default log level to WARNING to reduce verbosity
logger.setLevel(logging.WARNING)


class AnthropicDiscovery(BaseDiscoveryProvider):
    """
    Discovery provider for Anthropic models using direct REST API calls.

    This provider uses the /v1/models endpoint to retrieve the available Anthropic models
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
            raise ModelDiscoveryError.for_provider(
                provider="anthropic",
                reason="API key is not set in config or environment",
            )
        self._base_url: Optional[str] = base_url or "https://api.anthropic.com"
        client_kwargs: Dict[str, Any] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url

        # Instantiate the Anthropic client for other potential API interactions
        self.client: Anthropic = Anthropic(**client_kwargs)

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch Anthropic model metadata using direct REST API call.

        This method calls the /v1/models endpoint directly using the requests library
        to get the list of available models, and standardizes them for the model registry.
        It falls back to hardcoded models if the API call fails. Uses simplified error
        handling and aggressive timeouts to prevent hanging.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping standardized model IDs to their metadata.
        """
        start_time = time.time()
        logger.info("Starting Anthropic model fetch via REST API...")

        try:
            # Setting headers for the API request
            headers = {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }

            # Defining API endpoint URL
            models_url = f"{self._base_url}/v1/models"

            # Make API request with aggressive timeouts to prevent hanging
            # Use very short timeouts to fail fast rather than hang
            logger.info(f"Calling Anthropic REST API: {models_url} with timeout=(2,5)")

            # Direct request with aggressive timeouts
            response = requests.get(
                models_url,
                headers=headers,
                timeout=(
                    2,
                    5,
                ),  # (connect_timeout, read_timeout) in seconds - fairly aggressive
            )

            # Raising error for non-success responses
            response.raise_for_status()

            # Check if we're taking too long already
            if (
                time.time() - start_time > 10
            ):  # Safety check - if we've spent over 10s already
                logger.warning(
                    "API responded but processing is taking too long, using fallbacks"
                )
                return self._get_fallback_models()

            # Parse response
            response_json = response.json()

            # The Anthropic API can return data in different formats
            # Sometimes it's a dictionary with a 'data' key containing a list of models
            # Sometimes it's a list of models directly
            if isinstance(response_json, dict) and "data" in response_json:
                models_data = response_json.get("data", [])
            elif isinstance(response_json, list):
                models_data = response_json
            else:
                # Fallback to empty list if unexpected format
                logger.warning(
                    f"Unexpected format from Anthropic API: {type(response_json).__name__}. Using fallback models."
                )
                models_data = []

            logger.info(
                f"Successfully retrieved {len(models_data)} Anthropic models via REST API"
            )

            # Process model data efficiently
            logger.info(f"Processing {len(models_data)} Anthropic models")
            standardized_models: Dict[str, Dict[str, Any]] = {}

            # Process with a reasonable model count limit as a safeguard
            model_count_limit = 50  # Reasonable upper limit, reduced from 100
            for model in models_data[:model_count_limit]:
                # Handle both dictionary models and object models
                if isinstance(model, dict):
                    raw_model_id = model.get("id", "")
                    model_object = model.get("object", "model")
                    display_name = model.get("display_name", "")
                    created_at = model.get("created_at", "")
                else:
                    # Handle case where model might be an object with attributes
                    raw_model_id = getattr(model, "id", "")
                    model_object = getattr(model, "object", "model")
                    display_name = getattr(model, "display_name", "")
                    created_at = getattr(model, "created_at", "")

                # Extracting base version without date for standardized model ID
                base_model_id = self._extract_base_model_id(raw_model_id)
                canonical_id: str = self._generate_model_id(base_model_id)
                standardized_models[canonical_id] = self._build_model_entry(
                    model_id=canonical_id,
                    model_data={
                        "id": raw_model_id,
                        "object": model_object,
                        "display_name": display_name,
                        "created_at": created_at,
                    },
                )

            # Return discovered models, even if empty
            if not standardized_models:
                logger.warning("No Anthropic models found - API discovery required")

            duration = time.time() - start_time
            logger.info(
                f"Anthropic model fetch completed in {duration:.2f}s, found {len(standardized_models)} models"
            )
            return standardized_models

        except requests.RequestException as req_err:
            logger.error("Error fetching Anthropic models via REST API: %s", req_err)
            logger.warning("No fallback models provided - API discovery required")
            return {}
        except Exception as unexpected_err:
            logger.exception(
                "Unexpected error fetching Anthropic models: %s", unexpected_err
            )
            logger.warning("No fallback models provided - API discovery required")
            return {}

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
        model_name = model_data.get("id", model_id.split(":")[-1])
        return {
            "id": model_id,  # Using consistent field names with other providers
            "name": model_name,
            "model_id": model_id,  # For tests expecting model_id
            "model_name": model_name,  # For tests expecting model_name
            "api_data": model_data,
        }

    def _extract_base_model_id(self, raw_model_id: str) -> str:
        """
        Extract the base model ID, removing only date suffixes if present.

        This function performs minimal normalization, primarily removing date
        suffixes to maintain consistency across model versions.

        Args:
            raw_model_id (str): The raw model identifier from the API.

        Returns:
            str: The model identifier without date suffix.
        """
        # Log for debugging
        logger.debug(f"Processing model ID: {raw_model_id}")

        # Simply remove date suffixes (YYYYMMDD format) if present
        import re

        date_pattern = r"(-\d{8})"
        base_id = re.sub(date_pattern, "", raw_model_id)

        return base_id

    # Fallback methods removed in favor of direct API discovery
