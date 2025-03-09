"""Network of Networks (NON) API for Ember.

This module provides a clean interface for working with the Network of Networks
pattern in Ember, offering composable building blocks for LLM application patterns.

Examples:
    # Creating a simple ensemble with a judge
    from ember.api import non
    
    ensemble = non.UniformEnsemble(
        num_units=3,
        model_name="openai:gpt-4o", 
        temperature=1.0
    )
    
    judge = non.JudgeSynthesis(model_name="anthropic:claude-3-opus")
    
    pipeline = non.Sequential(operators=[ensemble, judge])
    
    result = pipeline(inputs={"query": "What is the capital of France?"})
"""

# Import from the implementation
from ember.core.non import (
    # Operator patterns
    UniformEnsemble,     # Generate multiple responses with identical models
    MostCommon,          # Select most common answer from ensemble
    JudgeSynthesis,      # Use a judge to synthesize multiple responses
    Verifier,            # Verify answers for correctness
    Sequential,          # Chain operators in sequence
    VariedEnsemble,      # Generate responses with varied model configurations
    
    # Input/Output types
    EnsembleInputs,      # Inputs for ensemble operators
    JudgeSynthesisInputs, JudgeSynthesisOutputs,  # Judge I/O
    VerifierInputs, VerifierOutputs,              # Verifier I/O
    VariedEnsembleInputs, VariedEnsembleOutputs   # Varied Ensemble I/O
)

__all__ = [
    # Operator patterns
    "UniformEnsemble",
    "MostCommon",
    "JudgeSynthesis",
    "Verifier",
    "Sequential",
    "VariedEnsemble",
    
    # Input/Output types
    "EnsembleInputs",
    "JudgeSynthesisInputs", "JudgeSynthesisOutputs",
    "VerifierInputs", "VerifierOutputs",
    "VariedEnsembleInputs", "VariedEnsembleOutputs"
]