"""
Ember type system.

This module provides the core type definitions for Ember, including protocols,
type variables, and utility types to ensure type safety throughout the codebase.
"""

from ember.core.types.ember_model import EmberModel
from ember.core.types.protocols import EmberTyped, EmberSerializable, TypeInfo
from ember.core.types.type_vars import (
    T,
    K,
    V,
    InputT,
    OutputT,
    ModelT,
    ProviderT,
    ConfigT,
)
from ember.core.types.config_types import (
    ConfigManager,
    ModelConfigDict,
    EmberConfigDict,
)
from ember.core.types.xcs_types import (
    XCSNode,
    XCSGraph,
    XCSPlan,
    XCSNodeAttributes,
    XCSNodeResult,
)
from ember.core.types.type_check import (
    validate_type,
    validate_instance_attrs,
    type_check,
)

__all__ = [
    # Core model
    "EmberModel",
    # Protocols
    "EmberTyped",
    "EmberSerializable",
    "TypeInfo",
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
