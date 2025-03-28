"""
Ember: Compositional Framework for Compound AI Systems
=====================================================

Ember is a powerful, extensible Python framework for building and orchestrating
Compound AI Systems and "Networks of Networks" (NONs).

Core Features:
- Eager Execution by Default
- Parallel Graph Execution
- Composable Operators
- Extensible Registry System
- Enhanced JIT System
- Built-in Evaluation
- Powerful Data Handling
- Intuitive Model Access

For more information, visit https://pyember.org

Examples:
    # Import primary API modules
    import ember

    # Initialize model registry and service
    from ember.api.models import initialize_registry, create_model_service

    registry = initialize_registry(auto_discover=True)
    model_service = create_model_service(registry=registry)

    # Call a model
    response = model_service.invoke_model(
        model_id="openai:gpt-4",
        prompt="What's the capital of France?",
        temperature=0.7
    )

    # Load datasets directly
    from ember.api.data import datasets
    mmlu_data = datasets("mmlu")

    # Or use the dataset builder pattern
    from ember.api.data import DatasetBuilder
    dataset = DatasetBuilder().split("test").sample(100).build("mmlu")

    # Create Networks of Networks (NONs)
    from ember.api import non
    ensemble = non.UniformEnsemble(
        num_units=3,
        model_name="openai:gpt-4o"
    )

    # Optimize with XCS
    from ember.api import xcs
    @xcs.jit
    def optimized_fn(x):
        return complex_computation(x)
"""

from __future__ import annotations

import importlib.metadata
from typing import Dict, Optional, Union, Any, Callable

# Import primary API components - these are the only public interfaces
from ember.api import (
    data,  # Dataset access (datasets("mmlu"), etc.)
    models,  # Language model access (models.openai.gpt4, etc.)
    non,  # Network of Networks patterns (non.UniformEnsemble, etc.)
    operators,  # Operator registry (operators.get_operator(), etc.)
    xcs,  # Execution optimization (xcs.jit, etc.)
)

# Import necessary components for initialization
# NOTE: These imports are moved to their usage points to avoid circular imports
# The components will be imported when needed in functions like initialize_ember and init

# Version detection
try:
    __version__ = importlib.metadata.version("ember-ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

# Package metadata
_PACKAGE_METADATA = {
    "name": "ember-ai",
    "version": __version__,
    "description": "Compositional framework for building and orchestrating Compound AI Systems and Networks of Networks (NONs).",
}


def initialize_ember(
    config_path: Optional[str] = None,
    auto_discover: bool = True,
    force_discovery: bool = False,
    api_keys: Optional[Dict[str, str]] = None,
    env_prefix: str = "EMBER_",
    initialize_context: bool = True,
    verbose_logging: bool = False,
) -> Union["ModelRegistry", "EmberAppContext"]:
    """Initialize the Ember framework with a single, unified call.

    This function provides a high-level entry point for initializing all Ember
    components in a single operation, abstracting away the complexity of the
    underlying architecture. It intelligently coordinates between the configuration
    system, model registry, and application context.

    The initialization follows a clear, predictable sequence:
    1. Configuration loading and validation (with environment variable support)
    2. API key acquisition from multiple sources with proper precedence
    3. Model registry preparation with provider discovery
    4. Application context initialization (when requested)

    Args:
        config_path: Optional path to a YAML configuration file. If not provided,
                   standard locations will be searched and environment variables used.
        auto_discover: Whether to automatically discover available models from
                     connected provider APIs. Defaults to True.
        force_discovery: Force model discovery even if auto_discover is False.
                       Useful when you need an explicit refresh of available models.
        api_keys: Optional dictionary of provider API keys (e.g., {"openai": "sk-..."}).
                These take precedence over config file and environment variables.
        env_prefix: Prefix for environment variables to consider. Defaults to "EMBER_".
        initialize_context: Whether to initialize the global application context.
                          Set to False to only initialize the model registry.
        verbose_logging: Whether to use verbose logging. If False (default), reduces
                       verbosity for non-essential components like model discovery and HTTP libraries.

    Returns:
        If initialize_context is True: A fully initialized EmberAppContext containing
        all core services (recommended for most applications).
        If initialize_context is False: The initialized ModelRegistry only.

    Examples:
        # Full initialization with default settings
        app = initialize_ember()

        # Initialize with custom configuration
        app = initialize_ember(config_path="/path/to/config.yaml")

        # Initialize with explicit API keys
        app = initialize_ember(api_keys={"openai": "sk-...", "anthropic": "sk-..."})

        # Get only the model registry without initializing the global context
        registry = initialize_ember(initialize_context=False)
        model = registry.get_model("openai:gpt-4")
    """
    # Import here to avoid circular imports
    from ember.core.utils.logging import configure_logging
    from ember.core.config.manager import create_config_manager

    # 0. Configure logging first
    configure_logging(verbose=verbose_logging)

    # 1. Create the configuration manager with the provided config path
    config_manager = create_config_manager(config_path=config_path)

    # 2. Apply API keys if provided (highest precedence)
    if api_keys:
        for provider, api_key in api_keys.items():
            config_manager.set_provider_api_key(provider, api_key)

    # 3. Initialize the model registry
    registry = models.initialize_registry(
        config_manager=config_manager,
        auto_discover=auto_discover,
        force_discovery=force_discovery,
    )

    # 4. Initialize application context if requested
    if initialize_context:
        # Import here to avoid circular imports
        from ember.core.app_context import (
            EmberAppContext,
            EmberContext,
            create_ember_app,
        )

        app_context = create_ember_app(config_path=config_path, verbose=verbose_logging)
        # Set the unified ember context as global
        EmberContext.initialize(app_context=app_context)
        return app_context

    # Return just the registry if global context isn't needed
    return registry


def init(
    config: Optional[Union[Dict[str, Any], ConfigManager]] = None,
    usage_tracking: bool = False,
) -> Callable:
    """Initialize Ember and return a unified model service.

    This function provides a simple entry point for initializing Ember and accessing
    models directly through a callable service object, as shown in the README examples.

    Args:
        config: Optional configuration to override defaults. Can be a dictionary or a ConfigManager
        usage_tracking: Whether to enable cost/token tracking

    Returns:
        A model service that can be called directly with models and prompts

    Examples:
        # Simple usage
        service = init()
        response = service("openai:gpt-4o", "What is the capital of France?")

        # With usage tracking
        service = init(usage_tracking=True)
        response = service(models.ModelEnum.gpt_4o, "What is quantum computing?")
        usage = service.usage_service.get_total_usage()
    """
    from ember.api.models import initialize_registry, ModelService, UsageService

    # Initialize configuration if needed
    config_manager = None
    if isinstance(config, dict):
        config_manager = create_config_manager()
        for key, value in config.items():
            config_manager.set(key, value)
    elif isinstance(config, ConfigManager):
        config_manager = config

    # Initialize the registry with auto-discovery
    registry = initialize_registry(auto_discover=True, config_manager=config_manager)

    # Create usage service if tracking is enabled
    usage_service = UsageService() if usage_tracking else None

    # Create a model service that can be called directly
    service = ModelService(registry=registry, usage_service=usage_service)

    # Create a wrapper that allows direct calling with model ID and prompt
    def service_wrapper(model_id_or_enum, prompt, **kwargs):
        return service.invoke_model(model_id_or_enum, prompt, **kwargs)

    # Add service attributes to the wrapper
    service_wrapper.model_service = service
    service_wrapper.registry = registry
    if usage_service:
        service_wrapper.usage_service = usage_service

    return service_wrapper


# Public interface - only export the main API components
__all__ = [
    "models",  # Language model access
    "data",  # Dataset access
    "operators",  # Operator registry
    "non",  # Network of Networks patterns
    "xcs",  # Execution optimization
    "initialize_ember",  # Global initialization function
    "init",  # Simple initialization function (matches README examples)
    "configure_logging",  # Logging configuration utility
    "set_component_level",  # Fine-grained logging control
    "__version__",
]