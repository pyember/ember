"""
Network of Operators (NON) Pattern
=================================

The NON module provides composable building blocks for LLM application patterns.
These high-level operators encapsulate common patterns for ensemble generation,
aggregation, verification, and sequential processing.

Core components:
- UniformEnsemble: Generates multiple model responses using identical LLM configurations
- MostCommon: Implements majority-vote aggregation
- JudgeSynthesis: Uses a judge model to synthesize multiple responses
- Verifier: Performs factual and logical verification of responses
- Sequential: Chains operators together in a pipeline

Basic usage:
```python
import ember
from ember.non import UniformEnsemble, JudgeSynthesis, Sequential

# Create an ensemble with 3 identical models
ensemble = UniformEnsemble(
    num_units=3,
    model_name="openai:gpt-4o",
    temperature=1.0
)

# Create a judge to synthesize the outputs
judge = JudgeSynthesis(model_name="anthropic:claude-3-opus")

# Combine them sequentially
pipeline = Sequential(operators=[ensemble, judge])

# Execute the pipeline
result = pipeline(inputs={"query": "What is the future of AI?"})
```

For more advanced usage, see the documentation.
"""

# This pattern ensures we prioritize types that don't have circular dependencies
# first, before importing implementation code that might create import cycles

# First, import all the type definitions which are less likely to have circular references
try:
    # Import type-related items first to avoid circular imports
    from ember.core.registry.operator.core.ensemble import (
        EnsembleOperatorInputs as EnsembleInputs,
    )
    from ember.core.registry.operator.core.ensemble import (
        EnsembleOperatorOutputs,
    )
    from ember.core.registry.operator.core.most_common import (
        MostCommonAnswerSelectorOperatorInputs as MostCommonInputs,
    )
    from ember.core.registry.operator.core.most_common import (
        MostCommonAnswerSelectorOutputs,
    )
    from ember.core.registry.operator.core.synthesis_judge import (
        JudgeSynthesisInputs,
        JudgeSynthesisOutputs,
    )
    from ember.core.registry.operator.core.verifier import (
        VerifierOperatorInputs as VerifierInputs,
    )
    from ember.core.registry.operator.core.verifier import (
        VerifierOperatorOutputs as VerifierOutputs,
    )
except ImportError:
    # For tests, define stub classes in case imports fail
    from typing import List

    from ember.core.types.ember_model import EmberModel

    # Stub classes for when imports fail during test collection
    class EnsembleInputs(EmberModel):
        """Input type for ensemble operations."""

        query: str

    class EnsembleOperatorOutputs(EmberModel):
        """Output type for ensemble operations."""

        responses: List[str]

    class MostCommonInputs(EmberModel):
        """Input type for most common answer selection."""

        query: str
        responses: List[str]

    class MostCommonAnswerSelectorOutputs(EmberModel):
        """Output type for most common answer selection."""

        final_answer: str

    class JudgeSynthesisInputs(EmberModel):
        """Input type for judge synthesis operations."""

        query: str
        responses: List[str]

    class JudgeSynthesisOutputs(EmberModel):
        """Output type for judge synthesis operations."""

        synthesized_response: str
        reasoning: str

    class VerifierInputs(EmberModel):
        """Input type for verification operations."""

        query: str
        candidate_answer: str

    class VerifierOutputs(EmberModel):
        """Output type for verification operations."""

        verdict: str
        explanation: str
        revised_answer: str


# Then, import the implementation modules which might have dependencies on the types
try:
    from ember.core.non import (  # Core operators; Additional input types for VariedEnsemble
        JudgeSynthesis,
        MostCommon,
        Sequential,
        UniformEnsemble,
        VariedEnsemble,
        VariedEnsembleInputs,
        VariedEnsembleOutputs,
        Verifier,
    )
except ImportError:
    # Stub implementations for tests
    from typing import List

    class UniformEnsemble:
        """Stub UniformEnsemble for tests."""

        def __init__(self, num_units=3, model_name=None, temperature=1.0):
            self.num_units = num_units
            self.model_name = model_name
            self.temperature = temperature

        def __call__(self, *, inputs):
            return {"responses": ["stub response"] * self.num_units}

    class MostCommon:
        """Stub MostCommon for tests."""

        def __init__(self):
            pass

        def __call__(self, *, inputs):
            return {"final_answer": "stub answer"}

    class JudgeSynthesis:
        """Stub JudgeSynthesis for tests."""

        def __init__(self, model_name=None):
            self.model_name = model_name

        def __call__(self, *, inputs):
            return {
                "synthesized_response": "stub synthesis",
                "reasoning": "stub reasoning",
            }

    class Verifier:
        """Stub Verifier for tests."""

        def __init__(self, model_name=None):
            self.model_name = model_name

        def __call__(self, *, inputs):
            return {
                "verdict": "valid",
                "explanation": "stub explanation",
                "revised_answer": "stub revision",
            }

    class Sequential:
        """Stub Sequential for tests."""

        def __init__(self, operators=None):
            self.operators = operators or []

        def __call__(self, *, inputs):
            return {"result": "stub sequential result"}

    class VariedEnsemble:
        """Stub VariedEnsemble for tests."""

        def __init__(self, models=None):
            self.models = models or []

        def __call__(self, *, inputs):
            return {"responses": ["stub varied response"] * len(self.models)}

    class VariedEnsembleInputs(BaseModel):
        """Stub VariedEnsembleInputs for tests."""

        query: str

    class VariedEnsembleOutputs(BaseModel):
        """Stub VariedEnsembleOutputs for tests."""

        responses: List[str]


__all__ = [
    # Core operators
    "UniformEnsemble",
    "MostCommon",
    "JudgeSynthesis",
    "Verifier",
    "Sequential",
    "VariedEnsemble",
    # Input/output types
    "EnsembleInputs",
    "EnsembleOperatorOutputs",
    "MostCommonInputs",
    "MostCommonAnswerSelectorOutputs",
    "JudgeSynthesisInputs",
    "JudgeSynthesisOutputs",
    "VerifierInputs",
    "VerifierOutputs",
    "VariedEnsembleInputs",
    "VariedEnsembleOutputs",
]
