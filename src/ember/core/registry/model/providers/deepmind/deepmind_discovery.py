import logging
import os
from typing import Dict, Any, List, Optional

from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError,
)
from ember.core.app_context import get_app_context

logger = logging.getLogger(__name__)


class DeepmindDiscovery(BaseDiscoveryProvider):
    """Discovery provider for Google Gemini models.

    This provider fetches available models from the Google Generative AI service.
    Each model is prefixed with 'google:' and returned as a dictionary of model details.
    """

    def __init__(self) -> None:
        """Initialize the DeepMind/Google discovery provider."""
        self._api_key: Optional[str] = None
        self._initialized: bool = False

    def configure(self, api_key: str) -> None:
        """Configure the discovery provider with API credentials.

        Args:
            api_key: The Google API key for authentication
        """
        self._api_key = api_key
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize the GenAI client with the API key.

        Raises:
            ModelDiscoveryError: If API key is missing or initialization fails
        """
        if self._initialized:
            return

        # Try to get API key from configuration if not provided
        if not self._api_key:
            try:
                # Get from centralized config
                app_context = get_app_context()
                provider_config = app_context.config_manager.get_config().get_provider(
                    "google"
                )
                if provider_config and provider_config.api_keys.get("default"):
                    self._api_key = provider_config.api_keys["default"].key
            except Exception as config_error:
                logger.debug(f"Could not get API key from config: {config_error}")

            # Fallback to environment variable
            if not self._api_key:
                self._api_key = os.environ.get("GOOGLE_API_KEY")

            if not self._api_key:
                raise ModelDiscoveryError(
                    "Google API key is not set in config or environment"
                )

        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._initialized = True
        except ImportError:
            logger.error("Google GenerativeAI SDK not installed")
            raise ModelDiscoveryError("Google GenerativeAI SDK not installed")
        except Exception as e:
            logger.exception(f"Error initializing Google client: {e}")
            raise ModelDiscoveryError(f"Failed to initialize Google client: {e}")

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch models available from Google Gemini and structure them for the registry.

        Returns:
            A dictionary where the keys are model IDs (prefixed with 'google:') and the
            values are dictionaries containing:
                - 'model_id': The unique model identifier.
                - 'model_name': The unmodified model name.
                - 'api_data': The raw API data returned for the model.

        Raises:
            ModelDiscoveryError: If the API key is missing or API access fails
        """
        try:
            # Ensure we're initialized with an API key
            self._initialize()

            import google.generativeai as genai

            models: Dict[str, Dict[str, Any]] = {}
            available_models: List[Any] = list(genai.list_models())

            for model in available_models:
                model_id: str = f"google:{model.name}"
                models[model_id] = {
                    "model_id": model_id,
                    "model_name": model.name,
                    "api_data": model,
                }

            if not models:
                # Add fallback models if API discovery returned empty
                return self._get_fallback_models()

            return models

        except ModelDiscoveryError:
            # Re-raise discovery errors for proper handling
            raise
        except Exception as error:
            logger.exception(f"Failed to fetch models from Google Gemini: {error}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve fallback models when API discovery fails.

        Returns:
            Dict containing fallback model data
        """
        return {
            "google:gemini-pro": {
                "model_id": "google:gemini-pro",
                "model_name": "gemini-pro",
                "api_data": {"object": "model"},
            },
            "google:gemini-ultra": {
                "model_id": "google:gemini-ultra",
                "model_name": "gemini-ultra",
                "api_data": {"object": "model"},
            },
            "google:gemini-1.5-pro": {
                "model_id": "google:gemini-1.5-pro",
                "model_name": "gemini-1.5-pro",
                "api_data": {"object": "model"},
            },
        }
