"""
Model Registry Initialization Module

This module provides the integration between the centralized configuration system
and the model registry, handling the initialization process with clean error handling
and consistent logging.
"""

import logging
from typing import Dict, List, Optional, Set, Any

from ember.core.exceptions import EmberError
from ember.core.config.manager import ConfigManager, create_config_manager
from ember.core.config.schema import EmberConfig
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit


logger = logging.getLogger(__name__)


def _convert_model_config_to_model_info(
    model_id: str,
    provider_name: str,
    model_config: Any,
    provider_config: Any,
    api_key: str
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
    if hasattr(model_config, "cost"):
        cost = ModelCost(
            input_cost_per_thousand=getattr(model_config.cost, "input_cost_per_thousand", 0.0),
            output_cost_per_thousand=getattr(model_config.cost, "output_cost_per_thousand", 0.0)
        )
    else:
        cost = ModelCost()
    
    # Create rate limit object
    if hasattr(model_config, "rate_limit"):
        rate_limit = RateLimit(
            tokens_per_minute=getattr(model_config.rate_limit, "tokens_per_minute", 0),
            requests_per_minute=getattr(model_config.rate_limit, "requests_per_minute", 0)
        )
    else:
        rate_limit = RateLimit()
    
    # Create provider info with custom args
    provider_info = ProviderInfo(
        name=provider_name.capitalize(),
        default_api_key=api_key,
        base_url=getattr(provider_config, "base_url", None)
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
    
    # Create and return model info
    return ModelInfo(
        model_id=model_id,
        model_name=model_name,
        cost=cost,
        rate_limit=rate_limit,
        provider=provider_info,
        api_key=api_key
    )


def initialize_registry(
    config_path: Optional[str] = None,
    config_manager: Optional[ConfigManager] = None,
    auto_discover: Optional[bool] = None,
    force_discovery: bool = False
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
        elif hasattr(config, "model_registry") and hasattr(config.model_registry, "auto_register"):
            # Legacy schema support
            auto_register_enabled = config.model_registry.auto_register or force_discovery
        
        # Register models from configuration
        if auto_register_enabled:
            registered_models = []
            
            # Process each provider - handle both new and old schema
            providers_dict = {}
            if hasattr(config, "registry") and hasattr(config.registry, "providers"):
                providers_dict = config.registry.providers
            elif hasattr(config, "model_registry") and hasattr(config.model_registry, "providers"):
                # Legacy schema support
                providers_dict = config.model_registry.providers
            
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
                        api_key = provider_config.api_keys["default"].key
                
                if not api_key:
                    logger.warning(f"No API key found for {provider_name}, skipping model registration")
                    continue
                
                # Register models for this provider - handle both dict and list formats
                model_configs = []
                
                # Handle provider.models as a list (new schema)
                if hasattr(provider_config, "models") and isinstance(provider_config.models, list):
                    model_configs = [(None, m) for m in provider_config.models]
                # Handle provider.models as a dict (old schema)
                elif hasattr(provider_config, "models") and isinstance(provider_config.models, dict):
                    model_configs = list(provider_config.models.items())
                
                for model_key, model_config in model_configs:
                    try:
                        # Generate model ID
                        if hasattr(model_config, "id"):
                            model_id = model_config.id
                            if ":" not in model_id:
                                model_id = f"{provider_name}:{model_id}"
                        elif model_key is not None:
                            # Use the dictionary key as model name
                            model_id = f"{provider_name}:{model_key}"
                        elif hasattr(model_config, "name"):
                            # Fallback for schema changes
                            model_id = f"{provider_name}:{model_config.name}"
                        else:
                            # Cannot determine model ID
                            logger.warning(f"Cannot determine model ID for {provider_name} model, skipping")
                            continue
                        
                        # Skip already registered models
                        if registry.is_registered(model_id):
                            logger.debug(f"Model {model_id} already registered, skipping")
                            continue
                        
                        # Convert to ModelInfo and register
                        model_info = _convert_model_config_to_model_info(
                            model_id=model_id,
                            provider_name=provider_name,
                            model_config=model_config,
                            provider_config=provider_config,
                            api_key=api_key
                        )
                        
                        registry.register_model(model_info)
                        registered_models.append(model_id)
                        logger.info(f"Registered model from config: {model_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to register model {getattr(model_config, 'id', 'unknown')}: {e}")
                
            if registered_models:
                logger.info(f"Registered {len(registered_models)} models from configuration")
            else:
                logger.info("No models registered from configuration")
        
        # Run model discovery if enabled or forced
        if discovery_enabled or force_discovery:
            logger.info("Running model discovery...")
            try:
                newly_discovered = registry.discover_models()
                if newly_discovered:
                    logger.info(f"Discovered {len(newly_discovered)} new models: {newly_discovered}")
                else:
                    logger.info("No new models discovered")
            except Exception as e:
                logger.error(f"Error during model discovery: {e}")
                
        return registry
        
    except Exception as e:
        logger.error(f"Error initializing model registry: {e}")
        raise EmberError(f"Failed to initialize model registry: {e}") from e