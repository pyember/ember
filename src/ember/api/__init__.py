"""Ember public API package.

This package provides a clean, stable public interface to Ember functionality.
All public APIs are accessible through this package, allowing users to import
from a single location while implementation details remain encapsulated.

Examples:
    # Import the main facades
    from ember.api import models, datasets, operators

    # Use models with a clean, namespaced interface
    response = models.openai.gpt4o("What's the capital of France?")

    # Load datasets directly
    mmlu_data = datasets("mmlu")

    # Or use the builder pattern
    from ember.api import DatasetBuilder
    dataset = DatasetBuilder().split("test").sample(100).build("mmlu")

    # Use Network of Networks (NON) patterns
    from ember.api import non
    ensemble = non.UniformEnsemble(num_units=3, model_name="openai:gpt-4o")

    # Optimize with XCS
    from ember.api import xcs
    @xcs.jit
    def optimized_fn(x):
        return complex_computation(x)
"""

# Import module namespaces
import ember.api.models as models  # Language model access
import ember.api.non as non  # Network of Networks patterns

# import ember.api.xcs as xcs  # Execution optimization
import ember.api.operators as operators  # Operator system

# Make operators available as both singular and plural for backward compatibility
operator = operators

# Import types API
import ember.api.types as types

# Import direct data API components
from ember.api.data import Dataset  # Dataset container class
from ember.api.data import DatasetBuilder  # Builder pattern for dataset configuration
from ember.api.data import DatasetConfig  # Configuration for dataset loading
from ember.api.data import DatasetEntry  # Individual dataset entry
from ember.api.data import DatasetInfo  # Dataset metadata
from ember.api.data import TaskType  # Enum of dataset task types
from ember.api.data import datasets  # Function to load datasets
from ember.api.data import get_dataset_info  # Get dataset metadata
from ember.api.data import list_available_datasets  # List available datasets
from ember.api.data import register  # Dataset registration decorator

# Import model API components
from ember.api.models import ModelAPI  # High-level model API
from ember.api.models import ModelBuilder  # Builder pattern for model configuration
from ember.api.models import ModelEnum  # Type-safe model references

# Public interface - export facades, modules, and direct API components
__all__ = [
    # Main facade objects
    "models",  # Model access (models.openai.gpt4o, etc.)
    "datasets",  # Dataset access (datasets("mmlu"), etc.)
    # Module namespaces
    "non",  # Network of Networks patterns
    # "xcs",  # Execution optimization
    "operators",  # Operator system (plural)
    "operator",  # Operator system (singular, for backward compatibility)
    "types",  # Types system for Ember models and operators
    # Model API components
    "ModelAPI",  # High-level model API
    "ModelBuilder",  # Builder pattern for model configuration
    "ModelEnum",  # Type-safe model references
    # Data API components
    "DatasetBuilder",  # Builder pattern for dataset loading
    "Dataset",  # Dataset container class
    "DatasetConfig",  # Configuration for dataset loading
    "TaskType",  # Enum of dataset task types
    "DatasetInfo",  # Dataset metadata
    "DatasetEntry",  # Individual dataset entry
    "register",  # Dataset registration decorator
    "list_available_datasets",  # List available datasets
    "get_dataset_info",  # Get dataset metadata
]
