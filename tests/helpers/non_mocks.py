"""
Production-quality mock implementations of Network of Networks (NON) components.

This module provides robust, well-tested mock implementations of the NON
patterns in Ember. These implementations follow proper interface contracts,
explicit type annotations, and error handling - making them suitable for both
testing and as fallbacks when full implementations are unavailable.

These mocks address the circular import issues in the production code while
maintaining the same interfaces.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Optional, TypeVar

from tests.helpers.ember_model import EmberModel
from tests.helpers.operator_base import Operator, Specification

# Setup logger
logger = logging.getLogger(__name__)

# Type variables for operator inputs/outputs
T_in = TypeVar("T_in", bound=EmberModel)
T_out = TypeVar("T_out", bound=EmberModel)

# -------------------------------------------------------------------------
# Mock Model Components
# -------------------------------------------------------------------------


class LMModuleConfig:
    """Configuration for LM modules."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize LM module configuration.

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens


class LMModule:
    """Mock implementation of a language model module."""

    def __init__(self, config: LMModuleConfig):
        """Initialize with configuration.

        Args:
            config: Configuration for the language model
        """
        self.config = config

    def __call__(self, *, prompt: str) -> str:
        """Execute the language model on a prompt.

        Args:
            prompt: The input prompt

        Returns:
            Generated text response
        """
        # For testing, just return a simple transformation of the prompt
        return (
            f"Response to: {prompt[:30]}..."
            if len(prompt) > 30
            else f"Response to: {prompt}"
        )


# -------------------------------------------------------------------------
# Ensemble Components
# -------------------------------------------------------------------------


class EnsembleOperatorInputs(EmberModel):
    """Inputs for ensemble operators."""

    query: str


class EnsembleOperatorOutputs(EmberModel):
    """Outputs from ensemble operators."""

    responses: List[str]


class EnsembleSpecification(Specification):
    """Specification for ensemble operations."""

    def __init__(self):
        """Initialize with appropriate models."""
        super().__init__(
            input_model=EnsembleOperatorInputs,
            structured_output=EnsembleOperatorOutputs,
        )


class UniformEnsemble(Operator[EnsembleOperatorInputs, EnsembleOperatorOutputs]):
    """Mock implementation of the UniformEnsemble operator."""

    specification: ClassVar[Specification] = EnsembleSpecification()

    def __init__(
        self,
        *,
        num_units: int,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the ensemble.

        Args:
            num_units: Number of model instances to use
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.num_units = num_units
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Create LM modules
        self.lm_modules = [
            LMModule(
                config=LMModuleConfig(
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            for _ in range(num_units)
        ]

    def forward(self, *, inputs: EnsembleOperatorInputs) -> EnsembleOperatorOutputs:
        """Execute the ensemble.

        Args:
            inputs: Input data containing the query

        Returns:
            Output data containing the responses
        """
        # Render prompt
        query = inputs.query

        # Execute on all models
        responses = [lm(prompt=query) for lm in self.lm_modules]

        return EnsembleOperatorOutputs(responses=responses)


# -------------------------------------------------------------------------
# MostCommon Components
# -------------------------------------------------------------------------


class MostCommonInputs(EmberModel):
    """Inputs for MostCommon operator."""

    responses: List[str]


class MostCommonOutputs(EmberModel):
    """Outputs from MostCommon operator."""

    most_common: str
    counts: Dict[str, int]


class MostCommonSpecification(Specification):
    """Specification for MostCommon operations."""

    def __init__(self):
        """Initialize with appropriate models."""
        super().__init__(
            input_model=MostCommonInputs, structured_output=MostCommonOutputs
        )


class MostCommon(Operator[MostCommonInputs, MostCommonOutputs]):
    """Mock implementation of the MostCommon operator."""

    specification: ClassVar[Specification] = MostCommonSpecification()

    def forward(self, *, inputs: MostCommonInputs) -> MostCommonOutputs:
        """Find the most common response.

        Args:
            inputs: Input data containing responses

        Returns:
            Output data with the most common response and counts
        """
        # Count occurrences
        counts: Dict[str, int] = {}
        for response in inputs.responses:
            counts[response] = counts.get(response, 0) + 1

        # Find most common
        most_common = max(counts.items(), key=lambda x: x[1])[0] if counts else ""

        return MostCommonOutputs(most_common=most_common, counts=counts)


# -------------------------------------------------------------------------
# Verifier Components
# -------------------------------------------------------------------------


class VerifierInputs(EmberModel):
    """Inputs for Verifier operator."""

    query: str
    candidate_answer: str


class VerifierOutputs(EmberModel):
    """Outputs from Verifier operator."""

    verdict: str
    explanation: str
    revised_answer: Optional[str] = None


class VerifierSpecification(Specification):
    """Specification for Verifier operations."""

    def __init__(self):
        """Initialize with appropriate models."""
        super().__init__(input_model=VerifierInputs, structured_output=VerifierOutputs)


class Verifier(Operator[VerifierInputs, VerifierOutputs]):
    """Mock implementation of the Verifier operator."""

    specification: ClassVar[Specification] = VerifierSpecification()

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the verifier.

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Create LM module
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
            )
        )

    def forward(self, *, inputs: VerifierInputs) -> VerifierOutputs:
        """Verify a candidate answer.

        Args:
            inputs: Input data containing query and candidate answer

        Returns:
            Output data with verification results
        """
        # For testing, simple verification logic
        if len(inputs.candidate_answer) < 10:
            verdict = "incorrect"
            explanation = "Answer is too short"
            revised_answer = f"Improved answer to: {inputs.query}"
        else:
            verdict = "correct"
            explanation = "Answer is adequate"
            revised_answer = None

        return VerifierOutputs(
            verdict=verdict, explanation=explanation, revised_answer=revised_answer
        )


# -------------------------------------------------------------------------
# JudgeSynthesis Components
# -------------------------------------------------------------------------


class JudgeSynthesisInputs(EmberModel):
    """Inputs for JudgeSynthesis operator."""

    query: str
    responses: List[str]


class JudgeSynthesisOutputs(EmberModel):
    """Outputs from JudgeSynthesis operator."""

    synthesized_response: str
    reasoning: str


class JudgeSynthesisSpecification(Specification):
    """Specification for JudgeSynthesis operations."""

    def __init__(self):
        """Initialize with appropriate models."""
        super().__init__(
            input_model=JudgeSynthesisInputs, structured_output=JudgeSynthesisOutputs
        )


class JudgeSynthesis(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Mock implementation of the JudgeSynthesis operator."""

    specification: ClassVar[Specification] = JudgeSynthesisSpecification()

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the judge.

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Create LM module
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
            )
        )

    def forward(self, *, inputs: JudgeSynthesisInputs) -> JudgeSynthesisOutputs:
        """Synthesize responses.

        Args:
            inputs: Input data containing query and responses

        Returns:
            Output data with synthesized response and reasoning
        """
        # For testing, simple synthesis logic
        query = inputs.query
        responses = inputs.responses

        synthesized = f"Synthesized answer to: {query}"
        reasoning = (
            f"Considered {len(responses)} responses and selected the best elements"
        )

        return JudgeSynthesisOutputs(
            synthesized_response=synthesized, reasoning=reasoning
        )


# -------------------------------------------------------------------------
# Sequential Component
# -------------------------------------------------------------------------


class Sequential(Operator[T_in, T_out]):
    """Mock implementation of the Sequential operator."""

    def __init__(self, *, operators: List[Operator[Any, Any]]):
        """Initialize with a list of operators.

        Args:
            operators: List of operators to execute sequentially
        """
        super().__init__()
        self.operators = operators

    def forward(self, *, inputs: T_in) -> T_out:
        """Execute operators sequentially.

        Args:
            inputs: Input data for the first operator

        Returns:
            Output data from the last operator
        """
        result = inputs
        for operator in self.operators:
            result = operator(inputs=result)
        return result


# Export all components
__all__ = [
    "LMModuleConfig",
    "LMModule",
    "EnsembleOperatorInputs",
    "EnsembleOperatorOutputs",
    "UniformEnsemble",
    "MostCommonInputs",
    "MostCommonOutputs",
    "MostCommon",
    "VerifierInputs",
    "VerifierOutputs",
    "Verifier",
    "JudgeSynthesisInputs",
    "JudgeSynthesisOutputs",
    "JudgeSynthesis",
    "Sequential",
]
