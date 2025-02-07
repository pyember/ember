import logging
from typing import Dict, Any, List
import google.generativeai as genai
from .base_discovery import BaseDiscoveryProvider

logger = logging.getLogger(__name__)


class GeminiDiscovery(BaseDiscoveryProvider):
    """Discovery provider for Google Gemini models.

    This provider fetches available models from the Google Generative AI service.
    Each model is prefixed with 'google:' and returned as a dictionary of model details.
    """

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch models available from Google Gemini and structure them for the registry.

        Returns:
            A dictionary where the keys are model IDs (prefixed with 'google:') and the
            values are dictionaries containing:
                - 'model_id': The unique model identifier.
                - 'model_name': The unmodified model name.
                - 'api_data': The raw API data returned for the model.
        """
        try:
            models: Dict[str, Dict[str, Any]] = {}
            available_models: List[Any] = list(genai.list_models())
            for model in available_models:
                model_id: str = f"google:{model.name}"
                models[model_id] = {
                    "model_id": model_id,
                    "model_name": model.name,
                    "api_data": model,
                }
            return models
        except Exception as error:
            logger.exception("Failed to fetch models from Google Gemini.")
            return {}
