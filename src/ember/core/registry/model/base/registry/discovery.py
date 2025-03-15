import os
import time
import logging
import threading
from typing import Any, Dict, List

from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider
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
    with local configuration overrides. Results are cached (with thread-safe access) to
    reduce redundant network calls.
    """

    def __init__(self, ttl: int = 3600) -> None:
        """Initialize the ModelDiscoveryService.

        Args:
            ttl (int): Cache time-to-live in seconds. Defaults to 3600.
        """
        # Load discovery providers dynamically based on available API keys
        self.providers: List[BaseDiscoveryProvider] = self._initialize_providers()
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: float = 0.0
        self.ttl: int = ttl
        self._lock: threading.RLock = threading.RLock()  # Added reentrant lock for thread safety
        
    def _initialize_providers(self) -> List[BaseDiscoveryProvider]:
        """Initialize discovery providers based on available API keys.
        
        This method creates provider instances with appropriate configuration:
        1. Checks for required API keys for each provider
        2. Only initializes providers with valid credentials
        3. Configures providers with appropriate settings
        
        Returns:
            List[BaseDiscoveryProvider]: A list of initialized discovery providers.
        """
        from typing import Dict, Tuple, Optional, Type, Callable
        
        # Define provider configurations
        ProviderConfig = Tuple[Type[BaseDiscoveryProvider], str, Optional[Callable[[], Dict[str, str]]]]
        
        provider_configs: List[ProviderConfig] = [
            # Provider class, env var name, optional config function
            (OpenAIDiscovery, "OPENAI_API_KEY", lambda: {"api_key": os.environ.get("OPENAI_API_KEY", "")}),
            (AnthropicDiscovery, "ANTHROPIC_API_KEY", lambda: {"api_key": os.environ.get("ANTHROPIC_API_KEY", "")}),
            (DeepmindDiscovery, "GOOGLE_API_KEY", lambda: {"api_key": os.environ.get("GOOGLE_API_KEY", "")})
        ]
        
        # Initialize providers with available credentials
        providers: List[BaseDiscoveryProvider] = []
        
        for provider_class, env_var_name, config_fn in provider_configs:
            api_key = os.environ.get(env_var_name)
            if api_key:
                try:
                    # Initialize with configuration if available
                    if config_fn:
                        config = config_fn()
                        if hasattr(provider_class, "configure"):
                            # If provider has a configure method, use it
                            instance = provider_class()
                            instance.configure(**config)  # type: ignore
                            providers.append(instance)
                        else:
                            # Otherwise try to pass config to constructor
                            providers.append(provider_class(**config))  # type: ignore
                    else:
                        # Simple initialization without config
                        providers.append(provider_class())
                    
                    logger.debug(
                        "%s found, initialized %s successfully", 
                        env_var_name, 
                        provider_class.__name__
                    )
                except Exception as init_error:
                    logger.error(
                        "Failed to initialize %s: %s", 
                        provider_class.__name__, 
                        init_error
                    )
            else:
                logger.info(
                    "%s not found, skipping %s", 
                    env_var_name, 
                    provider_class.__name__
                )
        
        if not providers:
            logger.warning(
                "No API keys found for any providers. "
                "Set one of %s environment variables to enable discovery.",
                ", ".join(env_var for _, env_var, _ in provider_configs)
            )
        
        return providers

    def discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Discover models using registered providers, with caching based on TTL.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from model ID to its metadata.

        Raises:
            ModelDiscoveryError: If no models can be discovered due to provider errors.
        """
        with self._lock:  # Protects access to the internal cache
            current_time: float = time.time()
            # If the cached results are still valid (within TTL), return them
            if self._cache and (current_time - self._last_update) < self.ttl:
                logger.info("Returning cached discovery results.")
                return self._cache.copy()  # Return a copy to avoid external mutation

            aggregated_models: Dict[str, Dict[str, Any]] = {}
            errors: List[str] = []

            # Attempt to fetch models from each provider
            for provider in self.providers:
                try:
                    # Set a timeout for fetch_models to prevent hanging
                    import concurrent.futures
                    import signal
                    
                    # Define a function to call fetch_models with a timeout
                    def fetch_with_timeout():
                        return provider.fetch_models()
                    
                    # Use ThreadPoolExecutor with a timeout to prevent hanging
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(fetch_with_timeout)
                        try:
                            provider_models = future.result(timeout=30)  # 30 second timeout
                            aggregated_models.update(provider_models)
                        except concurrent.futures.TimeoutError:
                            logger.error(
                                "Timeout while fetching models from %s",
                                provider.__class__.__name__,
                            )
                            errors.append(f"{provider.__class__.__name__}: Timeout after 30 seconds")
                        
                except Exception as e:
                    errors.append(f"{provider.__class__.__name__}: {e}")
                    logger.error(
                        "Failed to fetch models from %s: %s",
                        provider.__class__.__name__,
                        e,
                    )

            # If no models found and we had errors, raise exception
            if not aggregated_models and errors:
                from ember.core.registry.model.providers.base_discovery import (
                    ModelDiscoveryError,
                )

                raise ModelDiscoveryError(
                    f"No models discovered. Errors: {'; '.join(errors)}"
                )

            # Store a copy of the new results in the cache
            self._cache = aggregated_models.copy()
            self._last_update = current_time
            logger.info("Discovered models: %s", list(aggregated_models.keys()))
            return aggregated_models.copy()

    def merge_with_config(
        self, discovered: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ModelInfo]:
        """Merge discovered model metadata with local configuration overrides.

        Local configuration (loaded from YAML) takes precedence over API-reported data.
        This method also ensures that environment variables are considered for API keys.

        Args:
            discovered: Discovered model metadata from providers.

        Returns:
            Dict[str, ModelInfo]: Mapping from model ID to merged ModelInfo objects.
        """
        import os
        from ember.core.registry.model.config.settings import EmberSettings

        settings = EmberSettings()
        
        # Get models from local configuration
        local_models: Dict[str, Dict[str, Any]] = {}
        
        # Check if registry has models attribute before accessing it
        if hasattr(settings.registry, 'models'):
            local_models = {
                model.id: model.model_dump() for model in settings.registry.models
            }

        # Map provider prefixes to their environment variable keys
        provider_api_keys: Dict[str, str] = {
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
            "google": os.environ.get("GOOGLE_API_KEY", ""),
            "deepmind": os.environ.get("GOOGLE_API_KEY", ""),  # Uses same key as Google
        }
        
        def get_provider_from_model_id(model_id: str) -> str:
            """Extract provider name from model ID."""
            if ":" in model_id:
                return model_id.split(":", 1)[0].lower()
            return "unknown"
        
        merged_models: Dict[str, ModelInfo] = {}
        
        for model_id, api_metadata in discovered.items():
            provider_name = get_provider_from_model_id(model_id)
            api_key = provider_api_keys.get(provider_name, "")
            
            if model_id in local_models:
                # Local configuration overrides API metadata except for API keys
                # which we take from environment if available.
                merged_data: Dict[str, Any] = {**api_metadata, **local_models[model_id]}
                
                # Override with environment API key if available and not explicitly set
                if api_key and not merged_data.get("provider", {}).get("default_api_key"):
                    if "provider" not in merged_data:
                        merged_data["provider"] = {}
                    if isinstance(merged_data["provider"], dict):
                        merged_data["provider"]["default_api_key"] = api_key
            else:
                # For discovered models not in local config, create reasonable defaults
                logger.warning(
                    "Model %s discovered via API but not in local config; using defaults with environment API key.",
                    model_id,
                )
                
                # Extract provider prefix from model ID
                provider_prefix = provider_name.capitalize()
                
                merged_data = {
                    "id": model_id,
                    "name": api_metadata.get("model_name", api_metadata.get("name", model_id.split(":")[-1])),
                    "cost": {
                        "input_cost_per_thousand": 0.0,
                        "output_cost_per_thousand": 0.0,
                    },
                    "rate_limit": {"tokens_per_minute": 0, "requests_per_minute": 0},
                    "provider": {
                        "name": provider_prefix,
                        "default_api_key": api_key,
                    },
                }
                
            # Validate model info and add to results if valid
            try:
                # Skip models without API keys - they can't be used anyway
                if not merged_data.get("provider", {}).get("default_api_key"):
                    logger.warning(
                        "Skipping model %s because no API key is available", 
                        model_id
                    )
                    continue
                    
                # Create ModelInfo instance
                model_info = ModelInfo(**merged_data)
                merged_models[model_id] = model_info
                logger.debug("Successfully merged model info for %s", model_id)
                
            except Exception as validation_error:
                logger.error(
                    "Failed to merge model info for %s: %s", 
                    model_id, 
                    validation_error
                )
                
        if not merged_models:
            logger.warning(
                "No valid models found after merging with configuration. "
                "Check that API keys are set in environment variables and model schemas are valid."
            )
            
        return merged_models

    def refresh(self) -> Dict[str, ModelInfo]:
        """Force a refresh of model discovery and merge with local configuration."""
        with self._lock:
            try:
                discovered: Dict[str, Dict[str, Any]] = self.discover_models()
                merged = self.merge_with_config(discovered=discovered)
                # Only update cache if discovery is successful
                self._cache = discovered.copy()
                self._last_update = time.time()
                return merged
            except Exception as e:
                logger.error("Failed to refresh model discovery: %s", e)
                # Return last known good cache if available
                return self._cache.copy() if self._cache else {}

    def invalidate_cache(self) -> None:
        """Manually invalidate the cache, forcing a refresh on next discovery."""
        with self._lock:
            self._cache.clear()
            self._last_update = 0.0
            logger.info("Cache invalidated; next discovery will fetch fresh data.")

    async def discover_models_async(self) -> Dict[str, Dict[str, Any]]:
        """Asynchronously discover models using registered providers, with caching based on TTL."""
        import asyncio

        current_time: float = time.time()
        with self._lock:
            if self._cache and (current_time - self._last_update) < self.ttl:
                logger.info("Returning cached discovery results.")
                return self._cache.copy()

        aggregated_models: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        async def fetch_from_provider(provider: BaseDiscoveryProvider) -> None:
            try:
                # Each fetch is done in a thread with timeout
                try:
                    # Create a task with a timeout
                    fetch_task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(None, provider.fetch_models)
                    )
                    provider_models = await asyncio.wait_for(fetch_task, timeout=30)  # 30 second timeout
                    
                    with self._lock:
                        aggregated_models.update(provider_models)
                except asyncio.TimeoutError:
                    error_msg = f"{provider.__class__.__name__}: Timeout after 30 seconds"
                    errors.append(error_msg)
                    logger.error(
                        "Timeout while fetching models from %s", provider.__class__.__name__
                    )
            except Exception as e:
                error_msg = f"{provider.__class__.__name__}: {e}"
                errors.append(error_msg)
                logger.error(
                    "Failed to fetch models from %s: %s", provider.__class__.__name__, e
                )

        tasks = [fetch_from_provider(provider) for provider in self.providers]
        await asyncio.gather(*tasks)

        if not aggregated_models and errors:
            from ember.core.registry.model.providers.base_discovery import (
                ModelDiscoveryError,
            )

            raise ModelDiscoveryError(
                f"No models discovered. Errors: {'; '.join(errors)}"
            )

        with self._lock:
            self._cache = aggregated_models.copy()
            self._last_update = current_time
            logger.info("Discovered models: %s", list(aggregated_models.keys()))
            return aggregated_models.copy()
