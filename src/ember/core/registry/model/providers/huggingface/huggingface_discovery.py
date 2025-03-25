"""
HuggingFace model discovery provider.

This module implements model discovery using the Hugging Face Hub API.
It queries the Hub for available models with text generation capabilities,
then filters and standardizes them for the Ember model registry.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi

from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError,
)

# Module-level logger.
logger = logging.getLogger(__name__)
# Set default log level to WARNING to reduce verbosity
logger.setLevel(logging.WARNING)


class HuggingFaceDiscovery(BaseDiscoveryProvider):
    """Discovery provider for Hugging Face models.

    Retrieves models from the Hugging Face Hub that support text generation.
    Filters and formats them for use in the Ember model registry.
    """

    def __init__(self) -> None:
        """Initialize the HuggingFace discovery provider."""
        self._api_token: Optional[str] = None
        self._initialized: bool = False
        self._api = None

    def configure(self, api_token: str) -> None:
        """Configure the discovery provider with API credentials.

        Args:
            api_token: The Hugging Face API token for authentication.
        """
        self._api_token = api_token
        self._initialized = False
        self._api = HfApi(token=api_token)

    def discover(self) -> Dict[str, Dict[str, Any]]:
        """Discover available models from the Hugging Face Hub.

        Queries the Hub for popular models that support text generation,
        then formats them for use in the Ember model registry.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping model IDs to their details.

        Raises:
            ModelDiscoveryError: If discovery fails due to API errors or invalid credentials.
        """
        if not self._api_token:
            logger.warning("HuggingFace API token not provided, discovery limited")
            return {}

        if not self._initialized:
            logger.info("Initializing HuggingFace model discovery")
            self._initialized = True

        try:
            # Query for popular text generation models
            models = {}
            
            # Get featured models for text generation
            logger.info("Discovering featured text generation models from HuggingFace Hub")
            featured_models = self._api.list_models(
                filter="text-generation", 
                sort="downloads", 
                limit=20
            )
            
            # Process the discovered models
            for model in featured_models:
                model_id = model.id
                model_name = f"huggingface:{model_id}"
                
                # Get additional model info
                try:
                    model_info = self._api.model_info(model_id)
                    # Extract relevant metadata
                    models[model_name] = {
                        "id": model_name,
                        "name": model_id,
                        "display_name": model_id.split('/')[-1],
                        "capabilities": ["chat", "completion"],
                        "description": model_info.description or f"HuggingFace model: {model_id}",
                        "context_window": 4096,  # Default, would ideally query model card
                        "cost": {
                            "input_cost_per_thousand": 0.0,  # Adjust based on your pricing
                            "output_cost_per_thousand": 0.0,  # Adjust based on your pricing
                        }
                    }
                except Exception as model_err:
                    logger.debug(f"Could not fetch details for model {model_id}: {model_err}")
            
            logger.info(f"Discovered {len(models)} HuggingFace models")
            return models
            
        except Exception as exc:
            logger.exception("Error during HuggingFace model discovery: %s", exc)
            raise ModelDiscoveryError(f"HuggingFace model discovery failed: {exc}") 