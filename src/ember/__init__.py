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
from typing import Optional

# Import primary API components - these are the only public interfaces
from ember.api import (
    models,    # Language model access (models.openai.gpt4, etc.)
    data,  # Dataset access (datasets("mmlu"), etc.)
    operators, # Operator registry (operators.get_operator(), etc.)
    non,       # Network of Networks patterns (non.UniformEnsemble, etc.)
    xcs        # Execution optimization (xcs.jit, etc.)
)

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

# Public interface - only export the main API components
__all__ = [
    "models",    # Language model access
    "data",  # Dataset access
    "operators", # Operator registry
    "non",       # Network of Networks patterns
    "xcs",       # Execution optimization
    "__version__"
]
