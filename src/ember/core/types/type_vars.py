"""
Type variables for the Ember type system.

This module provides reusable TypeVar definitions with specific bounds
to ensure consistent type annotations throughout the codebase.
"""

from typing import TypeVar, Dict, Any

# Generic type variables
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# Ember-specific bounded type variables
from ember.core.types.ember_model import EmberModel

# Input/Output type variables for operators
InputT = TypeVar("InputT", bound=EmberModel)
OutputT = TypeVar("OutputT", bound=EmberModel)

# Provider and model type variables
ModelT = TypeVar("ModelT", bound="BaseModel")
ProviderT = TypeVar("ProviderT", bound="BaseProvider")

# Configuration type variables
ConfigT = TypeVar("ConfigT", bound=Dict[str, Any])
