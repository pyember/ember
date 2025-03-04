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

For more information, visit https://pyember.org
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING, Optional, Any, Union, Callable

if TYPE_CHECKING:
    from ember.core.registry.model import ModelRegistry, ModelService
    from ember.core.registry.model import initialize_ember

try:
    from .core.registry.model import initialize_ember, ModelRegistry, ModelService
    from .core.registry.model.base.services.usage_service import UsageService
except ImportError:
    # Alternative import path when the package is not properly installed
    from ember.core.registry.model import initialize_ember, ModelRegistry, ModelService
    from ember.core.registry.model.base.services.usage_service import UsageService

# Try to get version from package metadata, fallback to hardcoded version
try:
    __version__ = importlib.metadata.version("ember-ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["ModelRegistry", "ModelService", "initialize_ember", "init", "non", "jit", "autograph"]

# Package version and metadata
_PACKAGE_METADATA = {
    "name": "ember-ai",
    "version": __version__,
    "description": "Compositional framework for building and orchestrating Compound AI Systems and Networks of Networks (NONs).",
}


def __getattr__(name: str) -> object:
    """Lazy load main components using absolute imports."""
    if name == "ModelRegistry":
        try:
            from ember.core.registry.model.base.registry.model_registry import (
                ModelRegistry,
            )
            return ModelRegistry
        except ImportError:
            from ember.core.registry.model.base.registry.model_registry import (
                ModelRegistry,
            )
            return ModelRegistry
    if name == "ModelService":
        try:
            from ember.core.registry.model.base.services.model_service import (
                ModelService,
            )
            return ModelService
        except ImportError:
            from ember.core.registry.model.base.services.model_service import (
                ModelService,
            )
            return ModelService
    if name == "initialize_ember":
        try:
            from ember.core.registry.model import initialize_ember
            return initialize_ember
        except ImportError:
            from ember.core.registry.model import initialize_ember
            return initialize_ember
    if name == "non":
        try:
            from ember.core import non
            return non
        except ImportError:
            from ember.core import non
            return non
    if name == "jit":
        try:
            from ember.xcs.tracer import jit
            return jit
        except ImportError:
            from ember.xcs.tracer import jit
            return jit
    if name == "autograph":
        try:
            from ember.xcs.tracer import autograph
            return autograph
        except ImportError:
            from ember.xcs.tracer import autograph
            return autograph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def init(
    *,
    config_path: Optional[str] = None,
    auto_register: bool = True,
    auto_discover: bool = True,
    usage_tracking: bool = True,
) -> ModelService:
    """Initialize Ember with a single function call.
    
    This is the quickest way to get started with Ember. It:
    1. Sets up the model registry with available providers
    2. Enables usage tracking by default
    3. Returns a callable model service
    
    Args:
        config_path: Custom path to the YAML configuration file
        auto_register: Whether to automatically register local models
        auto_discover: Whether to automatically discover remote models
        usage_tracking: Whether to track token usage and costs (on by default)
        
    Returns:
        A callable model service for invoking language models
        
    Example:
        >>> import ember
        >>> service = ember.init()
        >>> response = service("openai:gpt-4o", "Hello world!")
        >>> print(response.data)
    """
    # Initialize the registry using the unified settings flow
    registry = initialize_ember(
        config_path=config_path,
        auto_register=auto_register,
        auto_discover=auto_discover,
    )

    # Create a UsageService if usage tracking is enabled (default: True)
    usage_service: Optional[UsageService] = UsageService() if usage_tracking else None

    # Return the callable ModelService
    return ModelService(registry=registry, usage_service=usage_service)
