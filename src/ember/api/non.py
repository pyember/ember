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
from ember.core.non import (  # Operator patterns; Input/Output types
    EnsembleInputs,  # Inputs for ensemble operators
    JudgeSynthesis,  # Use a judge to synthesize multiple responses
    JudgeSynthesisInputs,
    JudgeSynthesisOutputs,  # Judge I/O
    MostCommon,  # Select most common answer from ensemble
    Sequential,  # Chain operators in sequence
    UniformEnsemble,  # Generate multiple responses with identical models
    VariedEnsemble,  # Generate responses with varied model configurations
    VariedEnsembleInputs,
    VariedEnsembleOutputs,  # Varied Ensemble I/O
    Verifier,  # Verify answers for correctness
    VerifierInputs,
    VerifierOutputs,  # Verifier I/O
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
    "JudgeSynthesisInputs",
    "JudgeSynthesisOutputs",
    "VerifierInputs",
    "VerifierOutputs",
    "VariedEnsembleInputs",
    "VariedEnsembleOutputs",
]
