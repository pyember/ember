import time
import logging
import threading
from typing import Any, Dict, List

from src.ember.core.registry.model.base.schemas.model_info import ModelInfo
from src.ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider
from src.ember.core.registry.model.providers.openai.openai_discovery import (
    OpenAIDiscovery,
)
from src.ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)
from src.ember.core.registry.model.providers.deepmind.deepmind_discovery import (
    DeepmindDiscovery,
)

logger: logging.Logger = logging.getLogger(__name__)


class ModelDiscoveryService:
    """Service for aggregating and merging model metadata from various discovery providers.

    This service collects model information from different providers and merges the data
    with local configuration overrides. Results are cached (with thread-safe access) to
    reduce redundant network calls.
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
        self._lock: threading.RLock = (
            threading.RLock()
        )  # Added reentrant lock for thread safety

    def discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover models using registered providers, with caching based on TTL.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from model ID to its metadata.

        Raises:
            ModelDiscoveryError: If no models can be discovered due to provider errors.
        """
        with self._lock:
            current_time: float = time.time()
            if self._cache and (current_time - self._last_update) < self.ttl:
                logger.info("Returning cached discovery results.")
                return self._cache.copy()  # Return a copy to prevent external mutation

            aggregated_models: Dict[str, Dict[str, Any]] = {}
            errors: List[str] = []
            for provider in self.providers:
                try:
                    provider_models: Dict[str, Dict[str, Any]] = provider.fetch_models()
                    aggregated_models.update(provider_models)
                except Exception as e:
                    errors.append(f"{provider.__class__.__name__}: {e}")
                    logger.error(
                        "Failed to fetch models from %s: %s",
                        provider.__class__.__name__,
                        e,
                    )

            if not aggregated_models and errors:
                from src.ember.core.registry.model.providers.base_discovery import (
                    ModelDiscoveryError,
                )

                raise ModelDiscoveryError(
                    f"No models discovered. Errors: {'; '.join(errors)}"
                )

            self._cache = (
                aggregated_models.copy()
            )  # Store a copy to protect internal state
            self._last_update = current_time
            logger.info("Discovered models: %s", list(aggregated_models.keys()))
            return aggregated_models.copy()

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
        from src.ember.core.registry.model.config.settings import EmberSettings

        settings = EmberSettings()
        local_models: Dict[str, Dict[str, Any]] = {
            model.id: model.model_dump() for model in settings.registry.models
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
                    "id": model_id,
                    "name": api_metadata.get("name", model_id),
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
        with self._lock:  # Protect cache invalidation and discovery refresh
            self._cache.clear()  # Invalidate cache
            try:
                discovered: Dict[str, Dict[str, Any]] = self.discover_models()
                return self.merge_with_config(discovered=discovered)
            except Exception as e:
                self._logger.error("Failed to refresh model discovery: %s", e)
                raise
