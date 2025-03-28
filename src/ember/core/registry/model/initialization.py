"""
Model Registry Initialization Module

This module provides the integration between the centralized configuration system
and the model registry, handling the initialization process with clean error handling
and consistent logging.
"""

import logging
from typing import Any, Optional

from ember.core.config.manager import ConfigManager, create_config_manager
from ember.core.exceptions import EmberError
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo

logger = logging.getLogger(__name__)
# Set default log level to WARNING to reduce verbosity
logger.setLevel(logging.WARNING)


def _convert_model_config_to_model_info(
    model_id: str,
    provider_name: str,
    model_config: Any,
    provider_config: Any,
    api_key: str,
) -> ModelInfo:
    """
    Convert from configuration model to ModelInfo.

    Args:
        model_id: Full model identifier
        provider_name: Provider name
        model_config: Model configuration
        provider_config: Provider configuration
        api_key: API key to use

    Returns:
        ModelInfo instance ready for registration
    """
    # Extract model name from model_id or use name field
    if hasattr(model_config, "name"):
        model_name = model_config.name
    else:
        model_name = model_id.split(":")[-1]

    # Create cost object
    cost = ModelCost()
    if hasattr(model_config, "cost"):
        cost = ModelCost(
            input_cost_per_thousand=getattr(
                model_config.cost, "input_cost_per_thousand", 0.0
            ),
            output_cost_per_thousand=getattr(
                model_config.cost, "output_cost_per_thousand", 0.0
            ),
        )
    elif hasattr(model_config, "cost_input") and hasattr(model_config, "cost_output"):
        cost = ModelCost(
            input_cost_per_thousand=getattr(model_config, "cost_input", 0.0),
            output_cost_per_thousand=getattr(model_config, "cost_output", 0.0),
        )

    # Create rate limit object
    rate_limit = RateLimit()
    if hasattr(model_config, "rate_limit"):
        rate_limit = RateLimit(
            tokens_per_minute=getattr(model_config.rate_limit, "tokens_per_minute", 0),
            requests_per_minute=getattr(
                model_config.rate_limit, "requests_per_minute", 0
            ),
        )
    elif hasattr(model_config, "tokens_per_minute") or hasattr(
        model_config, "requests_per_minute"
    ):
        rate_limit = RateLimit(
            tokens_per_minute=getattr(model_config, "tokens_per_minute", 0),
            requests_per_minute=getattr(model_config, "requests_per_minute", 0),
        )

    # Create provider info with custom args
    provider_info = ProviderInfo(
        name=provider_name.capitalize(),
        default_api_key=api_key,
        base_url=getattr(provider_config, "base_url", None),
    )

    # Add additional provider parameters as custom args
    if hasattr(provider_config, "model_dump") and callable(provider_config.model_dump):
        try:
            exclude_fields = {"enabled", "api_keys", "models"}
            custom_args = provider_config.model_dump(exclude=exclude_fields)
            for key, value in custom_args.items():
                if key not in ["__root_key__"]:
                    provider_info.custom_args[key] = str(value)
        except Exception as e:
            logger.warning(f"Error extracting custom args: {e}")

    # Create and return model info with proper id field (not model_id)
    return ModelInfo(
        id=model_id,
        name=model_name,
        cost=cost,
        rate_limit=rate_limit,
        provider=provider_info,
        api_key=api_key,
    )


def initialize_registry(
    config_path: Optional[str] = None,
    config_manager: Optional[ConfigManager] = None,
    auto_discover: Optional[bool] = None,
    force_discovery: bool = False,
) -> ModelRegistry:
    """
    Initialize the model registry using the centralized configuration system.

    This function serves as the primary entry point for setting up the model registry
    with configuration-driven model registration and optional discovery.

    Args:
        config_path: Path to configuration file
        config_manager: Existing ConfigManager instance to use
        auto_discover: Override auto_discover setting from config
        force_discovery: Force model discovery even if auto_discover is False

    Returns:
        Initialized model registry with registered models

    Raises:
        EmberError: If initialization fails
    """
    try:
        # Create registry
        registry = ModelRegistry(logger=logger)

        # Get configuration
        if config_manager is None:
            config_manager = create_config_manager(config_path=config_path)

        config = config_manager.get_config()

        # Get auto-discover setting from config or override
        discovery_enabled = (
            auto_discover
            if auto_discover is not None
            else config.registry.auto_discover
        )

        # Check for auto_register in config (may not exist in new schema)
        auto_register_enabled = force_discovery
        if hasattr(config.registry, "auto_register"):
            auto_register_enabled = config.registry.auto_register or force_discovery
        elif hasattr(config, "model_registry") and hasattr(
            config.model_registry, "auto_register"
        ):
            # Legacy schema support
            auto_register_enabled = (
                config.model_registry.auto_register or force_discovery
            )

        # Register models from configuration
        if auto_register_enabled:
            registered_models = []

            # Get providers from registry configuration
            providers_dict = {}
            if hasattr(config, "registry") and hasattr(config.registry, "providers"):
                providers_dict = config.registry.providers
            else:
                logger.warning("No providers found in registry configuration")

            # Process each provider
            for provider_name, provider_config in providers_dict.items():
                # Skip disabled providers
                if not provider_config.enabled:
                    logger.info(f"Provider {provider_name} is disabled, skipping")
                    continue

                # Get API key
                api_key = None
                if hasattr(provider_config, "api_keys"):
                    if "default" in provider_config.api_keys:
                        # Support both object and dict formats
                        default_key = provider_config.api_keys["default"]
                        if hasattr(default_key, "key"):
                            api_key = default_key.key
                        elif isinstance(default_key, dict) and "key" in default_key:
                            api_key = default_key["key"]

                if not api_key:
                    logger.warning(
                        f"No API key found for {provider_name}, skipping model registration"
                    )
                    continue

                # Extract model configurations
                model_configs = []

                # Support both dict and list formats for backward compatibility
                if hasattr(provider_config, "models"):
                    if isinstance(provider_config.models, dict):
                        model_configs = list(provider_config.models.items())
                    elif isinstance(provider_config.models, list):
                        model_configs = [(None, m) for m in provider_config.models]
                    else:
                        logger.warning(f"Unsupported model format for {provider_name}")
                else:
                    logger.warning(f"No models defined for {provider_name}")

                for model_key, model_config in model_configs:
                    try:
                        # Generate model ID using consistent format
                        if hasattr(model_config, "id"):
                            model_id = model_config.id
                            # Ensure provider prefix in model ID
                            if ":" not in model_id:
                                model_id = f"{provider_name}:{model_id}"
                        elif hasattr(model_config, "name"):
                            # Use model name if ID not available
                            model_id = f"{provider_name}:{model_config.name}"
                        else:
                            # Cannot determine model ID
                            logger.warning(
                                f"Cannot determine model ID for {provider_name} model, skipping"
                            )
                            continue

                        # Skip already registered models
                        if registry.is_registered(model_id):
                            logger.debug(
                                f"Model {model_id} already registered, skipping"
                            )
                            continue

                        # Convert to ModelInfo and register
                        model_info = _convert_model_config_to_model_info(
                            model_id=model_id,
                            provider_name=provider_name,
                            model_config=model_config,
                            provider_config=provider_config,
                            api_key=api_key,
                        )

                        registry.register_model(model_info)
                        registered_models.append(model_id)
                        logger.info(f"Registered model from config: {model_id}")

                    except Exception as e:
                        logger.error(
                            f"Failed to register model {getattr(model_config, 'id', 'unknown')}: {e}"
                        )

            if registered_models:
                logger.info(
                    f"Registered {len(registered_models)} models from configuration"
                )
            else:
                logger.info("No models registered from configuration")

        # Run model discovery if enabled or forced
        if discovery_enabled or force_discovery:
            logger.info(
                "Execute model discovery (timeout: 30 seconds per provider, running in parallel)"
            )
            try:
                import time

                start_time = time.time()
                newly_discovered = registry.discover_models()
                duration = time.time() - start_time

                if newly_discovered:
                    # Handle newly_discovered correctly, which is a list not a dict
                    logger.info(
                        f"Discovered {len(newly_discovered)} new models in {duration:.2f}s: {newly_discovered[:10]}"
                        + (
                            f" and {len(newly_discovered) - 10} more"
                            if len(newly_discovered) > 10
                            else ""
                        )
                    )
                else:
                    logger.info(
                        f"No new models discovered (discovery completed in {duration:.2f}s)"
                    )
            except Exception as e:
                logger.error(f"Error during model discovery: {e}")
                logger.info("Continuing with available models from configuration")

        return registry

    except Exception as e:
        logger.error(f"Error initializing model registry: {e}")
        raise EmberError(f"Failed to initialize model registry: {e}") from e
