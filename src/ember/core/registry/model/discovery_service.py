import time
import logging
from typing import Any, Dict, List

from .core.schemas.model_info import ModelInfo
from .provider_registry.base_discovery import BaseDiscoveryProvider
from .provider_registry.openai.openai_discovery import OpenAIDiscovery
from .provider_registry.anthropic.anthropic_discovery import AnthropicDiscovery
from .provider_registry.deepmind.deepmind_discovery import GeminiDiscovery

logger: logging.Logger = logging.getLogger(__name__)


class ModelDiscoveryService:
    """Service for aggregating and merging model metadata from external discovery providers.

    This service collects model metadata from various external discovery providers and merges
    the information with local configuration overrides. The results are cached based on a
    specified time-to-live (TTL) to reduce redundant external requests.

    Attributes:
        ttl (int): Time-to-live (in seconds) for the cached discovery results.
        auto_discover (bool): If False, external model discovery is disabled.
    """

    def __init__(self, ttl: int = 3600, auto_discover: bool = True) -> None:
        """Initialize the ModelDiscoveryService.

        Args:
            ttl (int): The TTL for the cached discovery results, in seconds.
            auto_discover (bool): Flag indicating whether automatic external discovery is enabled.
        """
        self.providers: List[BaseDiscoveryProvider] = [
            OpenAIDiscovery(),
            AnthropicDiscovery(),
            GeminiDiscovery(),
        ]
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: float = 0.0
        self.ttl: int = ttl
        self.auto_discover: bool = auto_discover

    def discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover models using registered providers with TTL-based caching.

        If auto discovery is disabled, the method returns the current cache. If the cached
        results are still valid based on the TTL, the cached data is returned; otherwise,
        fresh metadata is fetched from all providers.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping model IDs to their corresponding
                metadata from external discovery providers.
        """
        if not self.auto_discover:
            logger.info("auto_discover is disabled; skipping external model discovery.")
            return self._cache if self._cache else {}

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
        """Merge discovered metadata with local configuration overrides.

        Local configuration rules take precedence during the merge. This method attempts to
        instantiate a ModelInfo object for each model entry. If instantiation fails, an error is
        logged for that specific model.

        Args:
            discovered (Dict[str, Dict[str, Any]]): A dictionary where each key is a model ID and
                each value is a dictionary of metadata retrieved from external providers.

        Returns:
            Dict[str, ModelInfo]: A dictionary mapping model IDs to their merged ModelInfo objects.
        """
        merged_models: Dict[str, ModelInfo] = {}
        for model_id, api_metadata in discovered.items():
            merged_data: Dict[str, Any] = api_metadata.copy()
            merged_data.setdefault("model_id", model_id)
            try:
                merged_models[model_id] = ModelInfo(**merged_data)
            except Exception as error:
                logger.error("Failed to merge model info for %s: %s", model_id, error)
        return merged_models

    def refresh(self) -> Dict[str, ModelInfo]:
        """Force a refresh of the discovery cache and return updated model metadata.

        This method clears the current cache, retrieves fresh model metadata from the external
        providers, and merges the results with local configuration overrides.

        Returns:
            Dict[str, ModelInfo]: A dictionary mapping model IDs to their updated ModelInfo objects.
        """
        self._cache.clear()
        discovered: Dict[str, Dict[str, Any]] = self.discover_models()
        return self.merge_with_config(discovered)
