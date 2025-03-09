"""Ember public API package.

This package provides a clean, stable public interface to Ember functionality.
All public APIs are accessible through this package, allowing users to import
from a single location while implementation details remain encapsulated.

Examples:
    # Import the main facades
    from ember.api import models, datasets, operators
    
    # Use models with a clean, namespaced interface
    response = models.openai.gpt4("What's the capital of France?")
    
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
# import ember.api.non as non                      # Network of Networks patterns
# import ember.api.xcs as xcs                      # Execution optimization

# Import direct data API components
from ember.api.data import (
    DatasetBuilder,      # Builder pattern for dataset configuration
    Dataset,             # Dataset container class
    DatasetConfig,       # Configuration for dataset loading
    TaskType,            # Enum of dataset task types
    DatasetInfo,         # Dataset metadata
    DatasetEntry,        # Individual dataset entry
    register,            # Dataset registration decorator
    list_available_datasets,  # List available datasets
    get_dataset_info     # Get dataset metadata
)

# Public interface - export facades, modules, and direct API components
__all__ = [
    # Main facade objects
    # "models",                # Model access (models.openai.gpt4, etc.)
    "datasets",              # Dataset access (datasets("mmlu"), etc.)
    
    # Module namespaces
    # "non",                   # Network of Networks patterns
    # "xcs",                   # Execution optimization
    
    # Data API components
    "DatasetBuilder",        # Builder pattern for dataset loading
    "Dataset",               # Dataset container class
    "DatasetConfig",         # Configuration for dataset loading
    "TaskType",              # Enum of dataset task types
    "DatasetInfo",           # Dataset metadata
    "DatasetEntry",          # Individual dataset entry
    "register",              # Dataset registration decorator
    "list_available_datasets", # List available datasets
    "get_dataset_info"       # Get dataset metadata
]