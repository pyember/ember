import time
import logging
from typing import Any, Dict, List

from ember.core.registry.model.core.schemas.model_info import ModelInfo
from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
)
from ember.core.registry.model.providers.openai.openai_discovery import (
    OpenAIDiscovery,
)
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)
from ember.core.registry.model.providers.deepmind.deepmind_discovery import (
    DeepmindDiscovery,
)

logger: logging.Logger = logging.getLogger(__name__)


class ModelDiscoveryService:
    """Service for aggregating and merging model metadata from various discovery providers.

    This service collects model information from different providers and merges the data
    with local configuration overrides. Results are cached to reduce redundant network calls.
    """

    def __init__(self, ttl: int = 3600) -> None:
        """Initialize the ModelDiscoveryService.

        Args:
            ttl (int): Cache time-to-live in seconds. Defaults to 3600.
        """
        self.providers: List[BaseDiscoveryProvider] = [
            OpenAIDiscovery(),
            AnthropicDiscovery(),
            DeepmindDiscovery(),
        ]
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: float = 0.0
        self.ttl: int = ttl

    def discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover models using registered providers, with caching based on TTL.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from model ID to its metadata.
        """
        current_time: float = time.time()
        if self._cache and (current_time - self._last_update) < self.ttl:
            logger.info("Returning cached discovery results.")
            return self._cache

        aggregated_models: Dict[str, Dict[str, Any]] = {}
        for provider in self.providers:
            provider_models: Dict[str, Dict[str, Any]] = provider.fetch_models()
            aggregated_models.update(provider_models)

        self._cache = aggregated_models
        self._last_update = current_time
        logger.info("Discovered models: %s", list(aggregated_models.keys()))
        return aggregated_models

    def merge_with_config(
        self, discovered: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ModelInfo]:
        """Merge discovered model metadata with local configuration overrides.

        Local configuration (loaded from YAML) takes precedence over API-reported data.

        Args:
            discovered (Dict[str, Dict[str, Any]]): Discovered model metadata.

        Returns:
            Dict[str, ModelInfo]: Mapping from model ID to merged ModelInfo objects.
        """
        from ember.core.registry.model.config.registry_settings import emberSettings

        settings = emberSettings()
        local_models: Dict[str, Dict[str, Any]] = {
            model.model_id: model.dict() for model in settings.registry.models
        }

        merged_models: Dict[str, ModelInfo] = {}
        for model_id, api_metadata in discovered.items():
            if model_id in local_models:
                # Local configuration overrides API metadata.
                merged_data: Dict[str, Any] = {**api_metadata, **local_models[model_id]}
            else:
                logger.warning(
                    "Model %s discovered via API but not in local config; using defaults.",
                    model_id,
                )
                merged_data = {
                    "model_id": model_id,
                    "model_name": api_metadata.get("model_name", model_id),
                    "cost": {
                        "input_cost_per_thousand": 0.0,
                        "output_cost_per_thousand": 0.0,
                    },
                    "rate_limit": {"tokens_per_minute": 0, "requests_per_minute": 0},
                    "provider": {"name": model_id.split(":")[0]},
                    "api_key": None,
                }
            try:
                merged_models[model_id] = ModelInfo(**merged_data)
            except Exception as err:
                logger.error("Failed to merge model info for %s: %s", model_id, err)
        return merged_models

    def refresh(self) -> Dict[str, ModelInfo]:
        """Force a refresh of model discovery and merge with local configuration.

        Returns:
            Dict[str, ModelInfo]: Updated mapping from model IDs to ModelInfo objects.
        """
        self._cache.clear()  # Invalidate cache
        discovered: Dict[str, Dict[str, Any]] = self.discover_models()
        return self.merge_with_config(discovered=discovered)
