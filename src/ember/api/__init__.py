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
import ember.api.eval as eval  # Evaluation module
import ember.api.models as models  # Language model access
import ember.api.non as non  # Network of Networks patterns
import ember.api.operators as operators  # Operator system
import ember.api.types as types

# Make operators available as both singular and plural for backward compatibility
operator = operators

# Import high-level API components
from ember.api.data import (
    Dataset,  # Dataset container class
    DatasetBuilder,  # Builder pattern for dataset configuration
    DatasetConfig,  # Configuration for dataset loading
    DatasetEntry,  # Individual dataset entry
    DatasetInfo,  # Dataset metadata
    TaskType,  # Enum of dataset task types
    datasets,  # Function to load datasets
    get_dataset_info,  # Get dataset metadata
    list_available_datasets,  # List available datasets
    register,  # Dataset registration decorator
)
from ember.api.eval import (
    EvaluationPipeline,  # Pipeline for batch evaluation
    Evaluator,  # Evaluator for model outputs
)
from ember.api.models import (
    ModelAPI,  # High-level model API
    ModelBuilder,  # Builder pattern for model configuration
    ModelEnum,  # Type-safe model references
)

# Public interface - export facades, modules, and direct API components
__all__ = [
    # Main facade objects
    "models",  # Model access (models.openai.gpt4o, etc.)
    "datasets",  # Dataset access (datasets("mmlu"), etc.)
    # Module namespaces
    "eval",  # Evaluation module
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
    # Evaluation API components
    "Evaluator",  # Evaluator for model outputs
    "EvaluationPipeline",  # Pipeline for batch evaluation
]
