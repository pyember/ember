"""
NON wrapper module

This module provides strongly typed wrappers for Ember's built-in operators.
Since the base classes (EmberModule/Operator) already enforce immutability and
tree registration, these wrappers simply subclass Operator and initialize their
sub-operators in __post_init__.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

# Ember package imports
try:
    # Try standard import path first (for installed package)
    from ember.core.registry.operator.base._module import ember_field
    from ember.core.registry.operator.base.operator_base import Operator, T_in, T_out
    from ember.core.registry.operator.core.ensemble import (
        EnsembleOperator,
        EnsembleOperatorInputs,
    )
    from ember.core.registry.operator.core.most_common import (
        MostCommonAnswerSelectorOperator,
        MostCommonAnswerSelectorOperatorInputs,
    )
    from ember.core.registry.operator.core.synthesis_judge import (
        JudgeSynthesisOperator,
        JudgeSynthesisInputs,
        JudgeSynthesisOutputs,
        JudgeSynthesisSignature,
    )
    from ember.core.registry.operator.core.verifier import (
        VerifierOperator,
        VerifierOperatorInputs,
        VerifierOperatorOutputs,
        VerifierSignature,
    )
    from ember.core.registry.prompt_signature.signatures import Signature
    from ember.core.registry.model.model_module.lm import LMModuleConfig, LMModule
except ImportError:
    # Fall back to src.ember path (for development)
    from ember.core.registry.operator.base._module import ember_field
    from ember.core.registry.operator.base.operator_base import Operator, T_in, T_out
    from ember.core.registry.operator.core.ensemble import (
        EnsembleOperator,
        EnsembleOperatorInputs,
    )
    from ember.core.registry.operator.core.most_common import (
        MostCommonAnswerSelectorOperator,
        MostCommonAnswerSelectorOperatorInputs,
    )
    from ember.core.registry.operator.core.synthesis_judge import (
        JudgeSynthesisOperator,
        JudgeSynthesisInputs,
        JudgeSynthesisOutputs,
        JudgeSynthesisSignature,
    )
    from ember.core.registry.operator.core.verifier import (
        VerifierOperator,
        VerifierOperatorInputs,
        VerifierOperatorOutputs,
        VerifierSignature,
    )
    from ember.core.registry.prompt_signature.signatures import Signature
    from ember.core.registry.model.model_module.lm import LMModuleConfig, LMModule

# Alias re-export types for backward compatibility with clients/tests from before our
# registry refactor.
from ember.core.registry.operator.core.ensemble import EnsembleOperatorOutputs
from ember.core.registry.operator.core.most_common import MostCommonAnswerSelectorOutputs

EnsembleInputs = EnsembleOperatorInputs
MostCommonInputs = MostCommonAnswerSelectorOperatorInputs
VerifierInputs = VerifierOperatorInputs
VerifierOutputs = VerifierOperatorOutputs

# ------------------------------------------------------------------------------
# 1) Ensemble Operator Wrapper
# ------------------------------------------------------------------------------


class UniformEnsemble(Operator[EnsembleInputs, EnsembleOperatorOutputs]):
    """Wrapper around EnsembleOperator for parallel LM module calls.

    Example:
        ensemble = UniformEnsemble(
            num_units=3,
            model_name="openai:gpt-4o",
            temperature=1.0
        )
        output = ensemble(inputs=EnsembleInputs(query="What is the capital of France?"))
        responses = output.responses
    """

    num_units: int
    model_name: str
    temperature: float
    max_tokens: Optional[int]

    signature: Signature = EnsembleOperator.signature

    ensemble_op: EnsembleOperator = ember_field(init=False)

    def __init__(
        self,
        *,
        num_units: int,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> None:
        # Normal, conventional __init__ assignments:
        self.num_units = num_units
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Construct LM modules based on the provided parameters.
        lm_modules: List[LMModule] = [
            LMModule(
                config=LMModuleConfig(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )
            for _ in range(self.num_units)
        ]
        # Use our helper to set the computed field.
        self.ensemble_op = EnsembleOperator(lm_modules=lm_modules)

    def forward(self, *, inputs: EnsembleInputs) -> EnsembleOperatorOutputs:
        """Delegates execution to the underlying EnsembleOperator."""
        return self.ensemble_op.forward(inputs=inputs)


# ------------------------------------------------------------------------------
# 2) MostCommon Operator Wrapper
# ------------------------------------------------------------------------------


class MostCommon(Operator[MostCommonInputs, MostCommonAnswerSelectorOutputs]):
    """Wrapper around MostCommonOperator for consensus selection.

    Example:
        aggregator = MostCommon()
        output = aggregator(inputs=MostCommonInputs(query="...", responses=["A", "B", "A"]))
        # output.final_answer will be "A"
    """

    signature: Signature = MostCommonAnswerSelectorOperator.signature
    most_common_op: MostCommonAnswerSelectorOperator = ember_field(init=False)

    def __init__(self) -> None:
        self.most_common_op = MostCommonAnswerSelectorOperator()

    def forward(self, *, inputs: MostCommonInputs) -> MostCommonAnswerSelectorOutputs:
        """Delegates execution to the underlying MostCommonOperator."""
        return self.most_common_op(inputs=inputs)


# ------------------------------------------------------------------------------
# 3) Judge Synthesis Operator Wrapper
# ------------------------------------------------------------------------------


class JudgeSynthesis(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Wrapper around JudgeSynthesisOperator for multi-response synthesis.

    Example:
        judge = JudgeSynthesis(model_name="gpt-4o")
        output = judge(inputs=JudgeSynthesisInputs(query="What is 2+2?", responses=["3", "4", "2"]))
    """

    signature: Signature = JudgeSynthesisSignature()
    model_name: str
    temperature: float
    max_tokens: Optional[int]
    judge_synthesis_op: JudgeSynthesisOperator = ember_field(init=False)

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        self._init_field(
            field_name="judge_synthesis_op",
            value=JudgeSynthesisOperator(lm_module=lm_module),
        )

    def forward(self, *, inputs: JudgeSynthesisInputs) -> JudgeSynthesisOutputs:
        """Delegates execution to the underlying JudgeSynthesisOperator."""
        return self.judge_synthesis_op.forward(inputs=inputs)


# ------------------------------------------------------------------------------
# 4) Verifier Operator Wrapper
# ------------------------------------------------------------------------------


class Verifier(Operator[VerifierInputs, VerifierOutputs]):
    """Wrapper around VerifierOperator to verify and potentially revise a candidate answer.

    Example:
        verifier = Verifier(model_name="gpt-4o")
        output = verifier(inputs=VerifierInputs(query="What is 2+2?", candidate_answer="5"))
        verdict = output.verdict
        explanation = output.explanation
        revised = output.revised_answer
    """

    signature: Signature = VerifierSignature()
    model_name: str
    temperature: float
    max_tokens: Optional[int]
    verifier_op: VerifierOperator

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        self.verifier_op = VerifierOperator(lm_module=lm_module)

    def forward(self, *, inputs: VerifierInputs) -> VerifierOutputs:
        """Delegates execution to the underlying VerifierOperator."""
        return self.verifier_op(inputs=inputs)


# ------------------------------------------------------------------------------
# 5) VariedEnsemble Operator Wrapper and Sequential Pipeline
# ------------------------------------------------------------------------------


class VariedEnsembleInputs(BaseModel):
    """Typed input for the VariedEnsemble operator.

    Attributes:
        query (str): The query to be processed across various model configurations.
    """

    query: str


class VariedEnsembleOutputs(BaseModel):
    """Typed output for the VariedEnsemble operator.

    Attributes:
        responses (List[str]): Collection of responses from different LM configurations.
    """

    responses: List[str]


class VariedEnsembleSignature(Signature):
    input_model: Type[BaseModel] = VariedEnsembleInputs
    output_model: Type[BaseModel] = VariedEnsembleOutputs


class VariedEnsemble(Operator[VariedEnsembleInputs, VariedEnsembleOutputs]):
    """Wrapper around multiple LM modules that runs varied configurations and aggregates outputs.

    Example:
        varied_ensemble = VariedEnsemble(model_configs=[config1, config2])
        outputs = varied_ensemble(inputs=VariedEnsembleInputs(query="Example query"))
    """

    signature: Signature = VariedEnsembleSignature()
    model_configs: List[LMModuleConfig]
    varied_ensemble_op: EnsembleOperator = ember_field(init=False)

    def __init__(self, *, model_configs: List[LMModuleConfig]) -> None:
        self.model_configs = model_configs
        self.lm_modules = tuple(LMModule(config=config) for config in model_configs)
        self.varied_ensemble_op = EnsembleOperator(
            lm_modules=tuple(LMModule(config=config) for config in model_configs)
        )

    def build_prompt(self, *, inputs: VariedEnsembleInputs) -> str:
        """Builds a prompt from the input model.

        If a prompt_template is defined in the signature, it is used; otherwise, defaults to the query.
        """
        if self.signature and self.signature.prompt_template:
            return self.signature.render_prompt(inputs=inputs)
        return str(inputs.query)

    def call_lm(self, *, prompt: str, lm: Any) -> str:
        """Call an LM module with a prompt.

        Args:
            prompt: The prompt to send to the LM
            lm: The LM module to call

        Returns:
            The LM's response as a string
        """
        return lm(prompt=prompt)

    def forward(self, *, inputs: VariedEnsembleInputs) -> VariedEnsembleOutputs:
        """Executes the varied ensemble operation and aggregates responses."""
        prompt = self.build_prompt(inputs=inputs)
        responses: List[str] = []
        for lm in self.lm_modules:
            response_text = self.call_lm(prompt=prompt, lm=lm).strip()
            responses.append(response_text)
        return VariedEnsembleOutputs(responses=responses)


class Sequential(Operator[T_in, T_out]):
    """Compositional operator that chains multiple operators sequentially.

    Example:
        pipeline = Sequential(operators=[op1, op2])
        final_output = pipeline(inputs={"value": 0})
    """

    operators: List[Operator[Any, Any]]
    signature: Signature = Signature(input_model=None, output_model=None)

    def __init__(self, *, operators: List[Operator[Any, Any]]) -> None:
        self.operators = operators

    def forward(self, *, inputs: T_in) -> T_out:
        """Executes the operators sequentially."""
        for op in self.operators:
            inputs = op(inputs=inputs)
        return inputs
