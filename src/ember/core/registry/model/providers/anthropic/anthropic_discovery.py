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

from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError,
)

# Module-level logger.
logger = logging.getLogger(__name__)


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
            raise ModelDiscoveryError(
                "Anthropic API key is not set in config or environment"
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
                timeout=(2, 5)  # (connect_timeout, read_timeout) in seconds - fairly aggressive
            )

            # Raising error for non-success responses
            response.raise_for_status()

            # Check if we're taking too long already
            if time.time() - start_time > 10:  # Safety check - if we've spent over 10s already
                logger.warning("API responded but processing is taking too long, using fallbacks")
                return self._get_fallback_models()
                
            # Parse response
            models_data = response.json().get("data", [])
            logger.info(
                f"Successfully retrieved {len(models_data)} Anthropic models via REST API"
            )

            # Process model data efficiently
            logger.info(f"Processing {len(models_data)} Anthropic models")
            standardized_models: Dict[str, Dict[str, Any]] = {}
            
            # Process with a reasonable model count limit as a safeguard
            model_count_limit = 50  # Reasonable upper limit, reduced from 100
            for model in models_data[:model_count_limit]:
                raw_model_id: str = model.get("id", "")
                # Extracting base version without date for standardized model ID
                base_model_id = self._extract_base_model_id(raw_model_id)
                canonical_id: str = self._generate_model_id(base_model_id)
                standardized_models[canonical_id] = self._build_model_entry(
                    model_id=canonical_id,
                    model_data={
                        "id": raw_model_id,
                        "object": model.get("object", "model"),
                        "display_name": model.get("display_name", ""),
                        "created_at": model.get("created_at", ""),
                    },
                )

            # Always add fallbacks if no models found
            if not standardized_models:
                logger.info("No Anthropic models found; adding fallback models")
                self._add_fallback_models(standardized_models)

            duration = time.time() - start_time
            logger.info(
                f"Anthropic model fetch completed in {duration:.2f}s, found {len(standardized_models)} models"
            )
            return standardized_models

        except requests.RequestException as req_err:
            logger.error("Error fetching Anthropic models via REST API: %s", req_err)
            logger.info("Using fallback models due to API request error")
            return self._get_fallback_models()
        except Exception as unexpected_err:
            logger.exception(
                "Unexpected error fetching Anthropic models: %s", unexpected_err
            )
            logger.info("Using fallback models due to unexpected error")
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

    def _extract_base_model_id(self, raw_model_id: str) -> str:
        """
        Extract the base model ID by removing any version suffix.

        Args:
            raw_model_id (str): The raw model identifier from the API.

        Returns:
            str: The extracted base model identifier.
        """
        # Log to help diagnose any issues with model ID extraction
        logger.debug(f"Extracting base model ID from: {raw_model_id}")

        # Strip date suffixes (YYYY-MM-DD format) if present
        import re

        # Match patterns like claude-3-sonnet-20240229 or claude-3-opus-20240229
        date_pattern = r"(-\d{8})"
        base_id = re.sub(date_pattern, "", raw_model_id)

        # Handle specific common cases
        if raw_model_id.startswith("claude-3-"):
            # Map claude-3-sonnet-YYYYMMDD to claude-3
            if any(
                model_type in raw_model_id for model_type in ["sonnet", "opus", "haiku"]
            ):
                return "claude-3"
        elif raw_model_id.startswith("claude-3.5-"):
            # Map claude-3.5-sonnet-YYYYMMDD to claude-3.5
            if any(model_type in raw_model_id for model_type in ["sonnet", "haiku"]):
                return "claude-3.5"
        elif raw_model_id.startswith("claude-3.7-"):
            # Map claude-3.7-sonnet-YYYYMMDD to claude-3.7
            if "sonnet" in raw_model_id:
                return "claude-3.7"
        elif raw_model_id == "claude-3":
            return "claude-3"
        elif raw_model_id == "claude-3.5":
            return "claude-3.5"
        elif raw_model_id == "claude-3.7":
            return "claude-3.7"

        # If we couldn't determine the base model, return the version without date
        return base_id

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
            "anthropic:claude-3-sonnet": {
                "model_id": "anthropic:claude-3-sonnet",
                "model_name": "claude-3-sonnet",
                "api_data": {
                    "id": "claude-3-sonnet",
                    "object": "model",
                    "display_name": "Claude 3 Sonnet",
                },
            },
            "anthropic:claude-3-opus": {
                "model_id": "anthropic:claude-3-opus",
                "model_name": "claude-3-opus",
                "api_data": {
                    "id": "claude-3-opus",
                    "object": "model",
                    "display_name": "Claude 3 Opus",
                },
            },
            "anthropic:claude-3-haiku": {
                "model_id": "anthropic:claude-3-haiku",
                "model_name": "claude-3-haiku",
                "api_data": {
                    "id": "claude-3-haiku",
                    "object": "model",
                    "display_name": "Claude 3 Haiku",
                },
            },
            "anthropic:claude-3.5-sonnet": {
                "model_id": "anthropic:claude-3.5-sonnet",
                "model_name": "claude-3.5-sonnet",
                "api_data": {
                    "id": "claude-3.5-sonnet",
                    "object": "model",
                    "display_name": "Claude 3.5 Sonnet",
                },
            },
            "anthropic:claude-3.7-sonnet": {
                "model_id": "anthropic:claude-3.7-sonnet",
                "model_name": "claude-3.7-sonnet",
                "api_data": {
                    "id": "claude-3.7-sonnet",
                    "object": "model",
                    "display_name": "Claude 3.7 Sonnet",
                },
            },
        }
