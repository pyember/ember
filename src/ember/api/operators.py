"""Operators API for Ember.

This module provides a unified interface for creating, composing, and executing operators
in the Ember framework. Operators are the fundamental computational units in Ember,
allowing for type-safe composition of language model operations.

Examples:
    # Creating a basic operator
    from ember.api.operators import Operator, EmberModel, Field

    class QuestionInput(EmberModel):
        question: str = Field(..., description="Question to answer")

    class AnswerOutput(EmberModel):
        answer: str = Field(..., description="Response to the question")

    class QuestionAnswerer(Operator[QuestionInput, AnswerOutput]):
        def forward(self, inputs: QuestionInput) -> AnswerOutput:
            # Process the input using a language model
            response = self.model.generate(inputs.question)
            return AnswerOutput(answer=response)

    # Using ensemble operators
    from ember.api.operators import EnsembleOperator

    ensemble = EnsembleOperator(
        operators=[
            QuestionAnswerer(model="gpt-4"),
            QuestionAnswerer(model="claude-3"),
        ]
    )

    # Using selector operators
    from ember.api.operators import MostCommonAnswerSelector

    pipeline = MostCommonAnswerSelector(
        operator=ensemble
    )

    # Execute the pipeline
    result = pipeline(QuestionInput(question="What is machine learning?"))
    print(result.answer)
"""

# Define improved type variables with explicit bounds and variance
from typing import TypeVar

# Core imports from operator base
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field

# Contravariant for inputs (accepts superclasses of the specified type)
# Covariant for outputs (accepts the specified type and its subclasses)
InputT = TypeVar("InputT", bound=EmberModel, contravariant=True)
OutputT = TypeVar("OutputT", bound=EmberModel, covariant=True)

# Re-export useful types
from typing import Any, Dict, List, Optional, TypeVar, Union

# Core operator implementations
from ember.core.registry.operator.core.ensemble import EnsembleOperator
from ember.core.registry.operator.core.most_common import (
    MostCommonAnswerSelectorOperator as MostCommonAnswerSelector,
)
from ember.core.registry.operator.core.selector_judge import SelectorJudgeOperator
from ember.core.registry.operator.core.synthesis_judge import JudgeSynthesisOperator
from ember.core.registry.operator.core.verifier import VerifierOperator

__all__ = [
    # Base classes
    "Operator",  # Base operator class for extension
    "Specification",  # Specification for operators
    "EmberModel",  # Base model class for inputs/outputs
    "Field",  # Field definition for model attributes
    "InputT",
    "OutputT",  # Type variables for generics
    # Built-in operators
    "EnsembleOperator",  # Runs multiple operators in parallel
    "MostCommonAnswerSelector",  # Selects most frequent answer
    "VerifierOperator",  # Verifies candidate answers
    "SelectorJudgeOperator",  # Selects best answer using a judge
    "JudgeSynthesisOperator",  # Synthesizes a response from multiple answers
    # Useful types
    "List",
    "Dict",
    "Any",
    "Optional",
    "Union",
    "TypeVar",
]
