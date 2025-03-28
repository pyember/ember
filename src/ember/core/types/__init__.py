"""
Ember type system.

Provides the core type definitions for Ember, including protocols, models,
type variables, and utility types to ensure type safety throughout the codebase.
This module centralizes and exports all type-related functionality.
"""

from ember.core.types.config_types import (
    ConfigManager,
    EmberConfigDict,
    ModelConfigDict,
)
from ember.core.types.ember_model import EmberModel
from ember.core.types.protocols import (
    EmberSerializable,  # Legacy alias
    EmberTyped,  # Legacy alias
    Serializable,
    TypedProtocol,
    TypeInfo,
)
from ember.core.types.type_check import (
    type_check,
    validate_instance_attrs,
    validate_type,
)
from ember.core.types.type_vars import (
    ConfigT,
    InputT,
    K,
    ModelT,
    OutputT,
    ProviderT,
    T,
    V,
)
from ember.core.types.xcs_types import (
    XCSGraph,
    XCSNode,
    XCSNodeAttributes,
    XCSNodeResult,
    XCSPlan,
)

__all__ = [
    # Core model
    "EmberModel",
    # Modern protocols
    "TypedProtocol",
    "Serializable",
    "TypeInfo",
    # Legacy protocol aliases
    "EmberTyped",
    "EmberSerializable",
    # Type variables
    "T",
    "K",
    "V",
    "InputT",
    "OutputT",
    "ModelT",
    "ProviderT",
    "ConfigT",
    # Config types
    "ConfigManager",
    "ModelConfigDict",
    "EmberConfigDict",
    # XCS types
    "XCSNode",
    "XCSGraph",
    "XCSPlan",
    "XCSNodeAttributes",
    "XCSNodeResult",
    # Type checking utilities
    "validate_type",
    "validate_instance_attrs",
    "type_check",
]
