"""Configuration module for Ember's model registry.

This module provides access to the centralized configuration system in ember.core.config.
"""

import logging
import warnings
from typing import Optional

from ember.core.config.manager import create_config_manager

# Import from current locations
# Import from centralized config system
from ember.core.config.schema import EmberConfig
from ember.core.exceptions import EmberError
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.initialization import initialize_registry

# Keep imports for compatibility with old code
try:
    from ember.core.config.exceptions import ConfigError
    from ember.core.config.loader import load_config
    from ember.core.config.schema import Model, Provider
except ImportError:
    # Fallbacks for environments without the old modules
    class Provider:
        pass

    class Model:
        pass

    class ConfigError(Exception):
        pass

    def load_config(config_path=None):
        return EmberConfig()


logger: logging.Logger = logging.getLogger(__name__)


# EmberSettings is now defined in ember.core.config.schema
# to avoid circular imports


def _get_provider_api_key(provider):
    """Extract API key from different provider schema versions.

    Args:
        provider: Provider configuration object

    Returns:
        API key string or None
    """
    # New schema uses api_keys.default.key
    if hasattr(provider, "api_keys") and provider.api_keys:
        try:
            return provider.api_keys.get("default", {}).get("key")
        except (AttributeError, TypeError):
            pass

    # Old schema uses api_key directly
    if hasattr(provider, "api_key"):
        return provider.api_key

    return None


def _register_provider_models(
    registry: ModelRegistry, provider_name: str, provider
) -> None:
    """Register models for a specific provider based on configuration.

    Args:
        registry: Model registry instance
        provider_name: Provider identifier
        provider: Provider configuration

    Raises:
        EmberError: If model registration fails
    """
    # Skip disabled providers (both schemas have this)
    if hasattr(provider, "enabled") and not provider.enabled:
        logger.info(f"Provider {provider_name} is disabled, skipping registration")
        return

    # Check API key
    api_key = _get_provider_api_key(provider)
    if not api_key:
        logger.warning(
            f"No API key found for provider {provider_name}, skipping model registration"
        )
        return

    # Get provider-specific configuration
    base_url = getattr(provider, "base_url", None)

    # Handle different ways of storing models
    if hasattr(provider, "models"):
        if isinstance(provider.models, dict):
            # Old schema: dict of model configs
            model_configs = provider.models.items()
        elif isinstance(provider.models, list):
            # New schema: list of model configs
            model_configs = [
                (m.id.split(":")[-1], m) for m in provider.models if hasattr(m, "id")
            ]
        else:
            logger.warning(f"Unsupported models format for provider {provider_name}")
            return
    else:
        logger.warning(f"No models found for provider {provider_name}")
        return

    # Register models
    for model_name, model_config in model_configs:
        try:
            # Create model ID
            model_id = f"{provider_name}:{model_name}"

            # Handle cost - different schemas
            if hasattr(model_config, "cost") and isinstance(
                model_config.cost, ModelCost
            ):
                # New schema has ModelCost object
                cost = model_config.cost
            else:
                # Old schema has cost_input/cost_output
                cost_input = getattr(model_config, "cost_input", 0.0)
                cost_output = getattr(model_config, "cost_output", 0.0)
                cost = ModelCost(
                    input_cost_per_thousand=cost_input,
                    output_cost_per_thousand=cost_output,
                )

            # Create rate limit config
            rate_limit = RateLimit(
                tokens_per_minute=getattr(model_config, "tokens_per_minute", 0),
                requests_per_minute=getattr(model_config, "requests_per_minute", 0),
            )

            # Get model name from different schemas
            if hasattr(model_config, "model_name"):
                model_name_str = model_config.model_name
            elif hasattr(model_config, "name"):
                model_name_str = model_config.name
            else:
                model_name_str = model_name

            # Create provider info with custom args from extended fields
            provider_info = ProviderInfo(
                name=provider_name.capitalize(),
                default_api_key=api_key,
                base_url=base_url,
            )

            # Try to add custom provider arguments if model_dump exists
            if hasattr(provider, "model_dump") and callable(provider.model_dump):
                try:
                    exclude_fields = {"enabled", "api_key", "api_keys", "models"}
                    for key, value in provider.model_dump(
                        exclude=exclude_fields
                    ).items():
                        if key not in ["__root_key__"]:
                            provider_info.custom_args[key] = str(value)
                except Exception as dump_error:
                    logger.debug(f"Error extracting custom args: {dump_error}")

            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                model_name=model_name_str,
                cost=cost,
                rate_limit=rate_limit,
                provider=provider_info,
                api_key=api_key,
            )

            # Register model if not already registered
            if not registry.is_registered(model_id):
                registry.register_model(model_info=model_info)
                logger.info(f"Registered model from config: {model_id}")
            else:
                logger.debug(f"Model {model_id} already registered, skipping")

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")


def _get_registry_config(config):
    """Extract registry configuration from config.

    Args:
        config: Configuration object

    Returns:
        Tuple of (registry_config, auto_discover_flag)
    """
    registry_config = None
    auto_discover = True  # Default value

    # Get registry configuration
    if hasattr(config, "registry"):
        registry_config = config.registry

        # Registry might be a dictionary in some test scenarios
        if isinstance(registry_config, dict):
            auto_discover = registry_config.get("auto_discover", True)
            # Handle dict.get("auto_register", ...) instead of .auto_register
            if "auto_register" in registry_config and registry_config["auto_register"]:
                logger.debug("Found auto_register in dictionary registry config")
        elif hasattr(registry_config, "auto_discover"):
            auto_discover = registry_config.auto_discover
        else:
            logger.debug("Found registry but no auto_discover attribute")
    else:
        # Default to empty registry with auto-discovery enabled
        logger.info("No registry configuration found, using defaults")
        return None, True

    logger.debug(f"Using registry with auto_discover={auto_discover}")
    return registry_config, auto_discover


def initialize_ember(
    config_path: Optional[str] = None,
    auto_discover: Optional[bool] = None,
    auto_register: Optional[bool] = None,
    force_discovery: bool = False,
) -> ModelRegistry:
    """Initialize Ember's model registry using the configuration system.

    DEPRECATED: This function is maintained for backward compatibility.
    New code should use initialize_registry() from ember.core.registry.model.initialization.

    Args:
        config_path: Custom path to configuration file
        auto_discover: Override auto_discover setting from config
        auto_register: Override auto_register setting from config
        force_discovery: Force model discovery even if auto_discover is False

    Returns:
        ModelRegistry: Fully configured model registry

    Raises:
        EmberError: If initialization fails
    """
    warnings.warn(
        "initialize_ember() is deprecated. Use initialize_registry() from "
        "ember.core.registry.model.initialization instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        # Use the centralized configuration system
        config_manager = create_config_manager(config_path=config_path)
        return initialize_registry(
            config_manager=config_manager,
            auto_discover=auto_discover,
            force_discovery=force_discovery,
        )
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise EmberError(f"Failed to initialize Ember: {e}") from e

    # The following implementation is kept for reference but is not used
    """
    registry = ModelRegistry(logger=logger)
    
    try:
        # Load configuration
        config = load_config(config_path=config_path)
        
        # Extract registry configuration
        registry_config, default_auto_discover = _get_registry_config(config)
        
        # Determine whether to run discovery
        discovery_enabled = auto_discover if auto_discover is not None else default_auto_discover
        
        # Register provider models if registry config is available
        if registry_config is not None:
            # Handle different registry config formats
            if isinstance(registry_config, dict):
                # Dictionary-based config (often from tests or raw YAML)
                if "models" in registry_config and isinstance(registry_config["models"], list):
                    # Format: registry.models = [model1, model2, ...]
                    for model in registry_config["models"]:
                        if isinstance(model, dict) and "id" in model and "provider" in model:
                            try:
                                # Extract provider name from model ID or provider dict
                                if ":" in model["id"]:
                                    provider_name = model["id"].split(":", 1)[0]
                                else:
                                    provider_name = model.get("provider", {}).get("name", "unknown")
                                
                                # Create model info directly
                                model_name = model["id"].split(":")[-1] if ":" in model["id"] else model["id"]
                                
                                cost = ModelCost(
                                    input_cost_per_thousand=model.get("cost", {}).get("input_cost_per_thousand", 0.0),
                                    output_cost_per_thousand=model.get("cost", {}).get("output_cost_per_thousand", 0.0)
                                )
                                
                                provider_dict = model.get("provider", {})
                                provider_info = ProviderInfo(
                                    name=provider_dict.get("name", provider_name),
                                    default_api_key=provider_dict.get("default_api_key", ""),
                                    base_url=provider_dict.get("base_url", None)
                                )
                                
                                # Create and register the model
                                model_info = ModelInfo(
                                    model_id=model["id"],
                                    model_name=model.get("name", model_name),
                                    cost=cost,
                                    rate_limit=RateLimit(
                                        tokens_per_minute=model.get("rate_limit", {}).get("tokens_per_minute", 0),
                                        requests_per_minute=model.get("rate_limit", {}).get("requests_per_minute", 0),
                                    ),
                                    provider=provider_info,
                                    api_key=model.get("api_key", provider_dict.get("default_api_key", ""))
                                )
                                
                                if not registry.is_registered(model["id"]):
                                    registry.register_model(model_info=model_info)
                                    logger.info(f"Registered model from list config: {model['id']}")
                            except Exception as e:
                                logger.error(f"Failed to register model from list: {e}")
                
                # Handle other dict formats
                elif "providers" in registry_config and isinstance(registry_config["providers"], dict):
                    # Format: registry.providers = {provider1: {...}, provider2: {...}}
                    for provider_name, provider in registry_config["providers"].items():
                        _register_provider_models(registry, provider_name, provider)
            
            # Object-based config (Pydantic models)
            elif hasattr(registry_config, "providers"):
                for provider_name, provider in registry_config.providers.items():
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
    """
