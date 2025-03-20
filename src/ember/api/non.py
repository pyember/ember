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
from ember.core.non import EnsembleInputs  # Inputs for ensemble operators
from ember.core.non import (
    JudgeSynthesis,  # Use a judge to synthesize multiple responses
)
from ember.core.non import JudgeSynthesisOutputs  # Judge I/O
from ember.core.non import MostCommon  # Select most common answer from ensemble
from ember.core.non import Sequential  # Chain operators in sequence
from ember.core.non import (
    UniformEnsemble,  # Generate multiple responses with identical models
)
from ember.core.non import (
    VariedEnsemble,  # Generate responses with varied model configurations
)
from ember.core.non import VariedEnsembleOutputs  # Varied Ensemble I/O
from ember.core.non import Verifier  # Verify answers for correctness
from ember.core.non import VerifierOutputs  # Verifier I/O
from ember.core.non import (  # Operator patterns; Input/Output types
    JudgeSynthesisInputs,
    VariedEnsembleInputs,
    VerifierInputs,
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
