"""High-performance thread-local context system for component discovery.

The Ember Context System provides a lightweight, zero-overhead approach 
to component discovery and dependency management with these core principles:

1. Thread isolation: Each thread maintains independent registries 
2. Zero-overhead access: Component lookups have minimal cost
3. Lazy initialization: Components are created only when needed
4. Self-registration: Components register themselves
5. Direct interaction: Components interact directly without indirection

This design eliminates circular dependencies, improves thread safety,
and enhances performance while significantly reducing complexity.

Typical usage:

    from ember.core.context.config import ConfigComponent
    from ember.core.context.model import ModelComponent
    from ember.core.context.data import DataComponent
    
    # Create core components
    config = ConfigComponent()
    model = ModelComponent()
    data = DataComponent()
    
    # Access components through their interfaces
    model_instance = model.get_model("anthropic:claude-3")
    
    # Use components directly
    result = model_instance.generate("Calculate 27*35")
"""

# Import components in alphabetical order
from .component import Component
from .config import ConfigComponent
from .data import DataComponent
from .ember_context import (
    EmberContext,
    current_context,
)  # Import for backward compatibility
from .management import (
    scoped_registry,
    seed_registry,
    temp_component,
    with_registry,
)
from .metrics import ComponentMetrics, MetricsComponent
from .model import ModelComponent
from .registry import Registry

__all__ = [
    # Core registry and component base
    "Registry",
    "Component",
    # Management utilities
    "scoped_registry",
    "temp_component",
    "with_registry",
    "seed_registry",
    # Core components
    "ConfigComponent",
    "DataComponent",
    "MetricsComponent",
    "ModelComponent",
    "ComponentMetrics",
    # Backward compatibility
    "EmberContext",
    "current_context",
]
