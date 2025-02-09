import logging
from typing import Dict, Any, List
import openai
from ..base_discovery import BaseDiscoveryProvider

logger = logging.getLogger(__name__)


class OpenAIDiscovery(BaseDiscoveryProvider):
    """Discovery provider for OpenAI models.

    Retrieves available models from the OpenAI API and converts them into a standardized format.
    """

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch available models from OpenAI.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping of standardized model IDs to model details.
        """
        try:
            response: Dict[str, Any] = openai.Model.list()
            logger.debug("Fetched OpenAI model list: %s", response)
            models: Dict[str, Dict[str, Any]] = {}
            model_list: List[Dict[str, Any]] = response.get("data", [])
            for model in model_list:
                raw_model_id: str = model.get("id", "")
                standardized_model_id: str = self._generate_model_id(
                    raw_model_id=raw_model_id
                )
                models[standardized_model_id] = self._build_model_entry(
                    model_id=standardized_model_id, model_data=model
                )
            return models
        except openai.error.OpenAIError as err:
            logger.exception(
                "Failed to fetch models from OpenAI due to an API error: %s", err
            )
            return {}

    def _generate_model_id(self, raw_model_id: str) -> str:
        """Generate a standardized model ID with the 'openai:' prefix.

        Args:
            raw_model_id (str): The raw model identifier from the API.

        Returns:
            str: The standardized model identifier, prefixed with 'openai:'.
        """
        return f"openai:{raw_model_id}"

    def _build_model_entry(
        self, model_id: str, model_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construct a standardized model entry.

        Args:
            model_id (str): The standardized model identifier.
            model_data (Dict[str, Any]): The raw model data from the OpenAI API.

        Returns:
            Dict[str, Any]: A dictionary containing model details.
        """
        return {
            "model_id": model_id,
            "model_name": model_data.get("id", ""),
            "api_data": model_data,
        }
