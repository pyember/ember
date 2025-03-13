"""Configuration module for Ember's model registry.

This module provides functionality for initializing and configuring the model
registry using the minimalist configuration system.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

from ember.core.exceptions import EmberError
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.config import load_config, EmberConfig, Provider, Model, ConfigError

logger: logging.Logger = logging.getLogger(__name__)


def _register_provider_models(
    registry: ModelRegistry,
    provider_name: str,
    provider: Provider
) -> None:
    """Register models for a specific provider based on configuration.
    
    Args:
        registry: Model registry instance
        provider_name: Provider identifier
        provider: Provider configuration
        
    Raises:
        EmberError: If model registration fails
    """
    # Skip disabled providers
    if not provider.enabled:
        logger.info(f"Provider {provider_name} is disabled, skipping registration")
        return
    
    # Check API key
    if not provider.api_key:
        logger.warning(f"No API key found for provider {provider_name}, skipping model registration")
        return
    
    # Get provider-specific configuration
    base_url = getattr(provider, "base_url", None)
    
    # Register models
    for model_name, model_config in provider.models.items():
        try:
            # Create model ID
            model_id = f"{provider_name}:{model_name}"
            
            # Create cost config
            cost = ModelCost(
                input_cost_per_thousand=model_config.cost_input,
                output_cost_per_thousand=model_config.cost_output
            )
            
            # Create rate limit config
            rate_limit = RateLimit(
                tokens_per_minute=getattr(model_config, "tokens_per_minute", 0),
                requests_per_minute=getattr(model_config, "requests_per_minute", 0)
            )
            
            # Create provider info with custom args from extended fields
            provider_info = ProviderInfo(
                name=provider_name.capitalize(),
                default_api_key=provider.api_key,
                base_url=base_url
            )
            
            # Add any custom provider arguments
            for key, value in provider.model_dump(exclude={"enabled", "api_key", "models"}).items():
                if key not in ["__root_key__"]:
                    provider_info.custom_args[key] = str(value)
            
            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                model_name=model_config.name,
                cost=cost,
                rate_limit=rate_limit,
                provider=provider_info,
                api_key=provider.api_key
            )
            
            # Register model if not already registered
            if not registry.is_registered(model_id):
                registry.register_model(model_info=model_info)
                logger.info(f"Registered model from config: {model_id}")
            else:
                logger.debug(f"Model {model_id} already registered, skipping")
                
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")


def initialize_ember(
    config_path: Optional[str] = None,
    auto_discover: Optional[bool] = None,
    force_discovery: bool = False,
) -> ModelRegistry:
    """Initialize Ember's model registry using the configuration system.
    
    This function is the recommended entry point for setting up Ember. It performs:
      - Loading configuration from files and environment variables
      - Creating and configuring the ModelRegistry
      - Registering models from configuration
      - Optional discovery of models from provider APIs
    
    Args:
        config_path: Custom path to configuration file
        auto_discover: Override auto_discover setting from config
        force_discovery: Force model discovery even if auto_discover is False
        
    Returns:
        ModelRegistry: Fully configured model registry
        
    Raises:
        EmberError: If initialization fails
    """
    try:
        # Load configuration
        config = load_config(file_path=config_path)
        
        # Override auto_discover if provided
        discovery_enabled = auto_discover if auto_discover is not None else config.registry.auto_discover
        
        # Create registry
        registry = ModelRegistry(logger=logger)
        
        # Register provider models from configuration
        for provider_name, provider in config.registry.providers.items():
            _register_provider_models(registry, provider_name, provider)
        
        # Run auto-discovery if enabled
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
        
    except ConfigError as e:
        logger.error(f"Configuration error during initialization: {e}")
        raise EmberError(f"Failed to initialize Ember: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        raise EmberError(f"Failed to initialize Ember: {e}") from e
