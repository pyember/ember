"""
Network of Operators (NON) Pattern Implementation

This module provides composable building blocks for LLM application patterns.
It wraps core operators with convenient, strongly-typed interfaces for common
LLM workflows like ensembles, majority voting, and verification.

Each wrapper maintains the immutability and functional approach of the underlying
system while offering simple, intuitive interfaces for application developers.

Example usage:
    ```python
    # Create an ensemble with 3 identical models
    ensemble = UniformEnsemble(
        num_units=3, 
        model_name="openai:gpt-4o",
        temperature=1.0
    )
    
    # Create a judge to synthesize the outputs
    judge = JudgeSynthesis(model_name="claude-3-opus")
    
    # Combine them sequentially
    pipeline = Sequential(operators=[ensemble, judge])
    
    # Execute the pipeline
    result = pipeline(inputs={"query": "What is the future of AI?"})
    ```
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
        JudgeSynthesisSpecification,
    )
    from ember.core.registry.operator.core.verifier import (
        VerifierOperator,
        VerifierOperatorInputs,
        VerifierOperatorOutputs,
        VerifierSpecification,
    )
    from ember.core.registry.prompt_specification.specification import Specification
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
        JudgeSynthesisSpecification,
    )
    from ember.core.registry.operator.core.verifier import (
        VerifierOperator,
        VerifierOperatorInputs,
        VerifierOperatorOutputs,
        VerifierSpecification,
    )
    from ember.core.registry.prompt_specification.specification import Specification
    from ember.core.registry.model.model_module.lm import LMModuleConfig, LMModule

# Alias re-export types for backward compatibility with clients/tests from before our
# registry refactor.
from ember.core.registry.operator.core.ensemble import EnsembleOperatorOutputs
from ember.core.registry.operator.core.most_common import (
    MostCommonAnswerSelectorOutputs,
)

EnsembleInputs = EnsembleOperatorInputs
MostCommonInputs = MostCommonAnswerSelectorOperatorInputs
VerifierInputs = VerifierOperatorInputs
VerifierOutputs = VerifierOperatorOutputs

# ------------------------------------------------------------------------------
# 1) Ensemble Operator Wrapper
# ------------------------------------------------------------------------------


class UniformEnsemble(Operator[EnsembleInputs, EnsembleOperatorOutputs]):
    """
    Generates multiple model responses using identical LLM configurations.
    
    Creates a set of parallel model instances with the same configuration, 
    sends the same prompt to each, and returns all responses. This helps
    mitigate non-determinism through statistical aggregation.
    
    Usage Notes:
    - Typically paired with MostCommon or JudgeSynthesis for aggregation
    - Higher num_units improves robustness but increases cost
    - Higher temperature increases response diversity
    
    Example:
        ensemble = UniformEnsemble(
            num_units=3,
            model_name="openai:gpt-4o",
            temperature=1.0
        )
        output = ensemble(inputs=EnsembleInputs(query="What is the capital of France?"))
        responses = output.responses  # List of 3 responses
    """

    num_units: int
    model_name: str
    temperature: float
    max_tokens: Optional[int]

    specification: Specification = EnsembleOperator.specification

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
    """
    Statistical consensus aggregator implementing a majority-vote decision strategy.
    
    MostCommon implements a robust, non-parametric approach to ensemble aggregation
    based on frequency counting. This operator identifies the most frequently occurring
    response in a collection, making it ideal for ensemble decision-making without
    introducing additional model-based bias.
    
    The algorithm employs a sophisticated frequency analysis that:
    1. Identifies exact matches through string equality
    2. Counts occurrence frequencies across all responses
    3. Selects the most frequent response as the consensus answer
    4. Handles ties deterministically (first occurrence wins)
    
    This approach offers several key advantages:
    - Model-agnostic aggregation (no additional inference needed)
    - Deterministic behavior with consistent outputs
    - Low computational overhead for high-performance workflows
    - Transparent decision mechanisms for interpretability
    
    The implementation abides by functional programming principles:
    - Pure function semantics with no side effects
    - Immutability of both the operator and processed data
    - Explicit input/output contract without hidden state
    
    Use this operator as the final stage in ensemble pipelines when:
    - You need statistical robustness against model hallucinations
    - Deterministic aggregation is more important than nuanced synthesis
    - The responses are expected to contain a clear majority answer
    
    Example:
        aggregator = MostCommon()
        output = aggregator(inputs=MostCommonInputs(
            query="What is 2+2?", 
            responses=["4", "4", "3", "4", "5"]
        ))
        # output.final_answer will be "4" (occurring 3 times)
    """

    specification: Specification = MostCommonAnswerSelectorOperator.specification
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
    """
    Intelligent meta-reasoning engine for synthesizing multiple model responses.
    
    JudgeSynthesis represents a sophisticated approach to ensemble aggregation that 
    leverages a "judge" LLM to analyze, evaluate, and synthesize multiple candidate 
    responses. Unlike statistical approaches, this operator applies reasoning capabilities
    to generate a superior answer that may incorporate elements from multiple responses
    or provide novel insights that resolve conflicts between them.
    
    The synthesis process follows a principled methodology:
    1. Candidate analysis - Each response is individually evaluated for quality and relevance
    2. Comparative assessment - Responses are compared and contrasted to identify patterns 
    3. Critical evaluation - Factual correctness and reasoning quality are assessed
    4. Reasoned synthesis - A new response is generated that represents the best integrated answer
    
    This approach implements a meta-learning paradigm where:
    - A higher-quality model can supervise and improve outputs from other models
    - Conflicting information across responses can be resolved through reasoning
    - The final answer can exceed the quality of any individual input response
    - The synthesis provides both a final answer and an explanation of the reasoning
    
    Implementation follows SOLID principles through:
    - Single Responsibility - Focused solely on response synthesis
    - Open/Closed - Extensible design with configurable model selection
    - Liskov Substitution - Properly typed interfaces enable seamless composition
    - Interface Segregation - Minimal, focused API for synthesis operations
    - Dependency Inversion - Configuration injected through constructor
    
    This pattern is ideal for mission-critical applications where:
    - Response quality and correctness are paramount concerns
    - You need reasoning-based aggregation rather than simple statistics
    - The task involves complex, nuanced judgments requiring critical thinking
    - A trace of meta-reasoning about the decision process is valuable
    
    Example:
        judge = JudgeSynthesis(model_name="anthropic:claude-3-opus")
        result = judge(inputs=JudgeSynthesisInputs(
            query="What is the impact of rising sea levels?",
            responses=[response1, response2, response3]
        ))
        final_answer = result.synthesized_response  # The reasoned synthesis
        reasoning = result.reasoning              # Explanation of the synthesis process
    """

    specification: Specification = JudgeSynthesisSpecification()
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
    """
    Autonomous verification and correction system for answer quality assurance.
    
    The Verifier operator implements a critical quality control mechanism for LLM outputs,
    functioning as an independent verification layer that scrutinizes candidate answers
    for correctness, completeness, and coherence. This pattern follows established
    software engineering principles of separation of concerns by isolating the verification
    process from answer generation.
    
    The verification process employs a structured methodology:
    1. Correctness assessment - Evaluates factual accuracy and logical consistency
    2. Error identification - Pinpoints specific issues in the candidate answer
    3. Detailed explanation - Provides comprehensive reasoning for the verdict
    4. Correction formulation - When errors are found, generates an improved answer
    
    This implementation offers several architectural advantages:
    - Decoupled verification logic from answer generation
    - Independent error detection and correction capabilities
    - Auditable decision process with explicit reasoning
    - Strong type safety with comprehensive validation
    
    The design follows robust software engineering principles:
    - Single Responsibility - Focused exclusively on verification and correction
    - Interface Segregation - Clean, minimal interface for verification operations
    - Dependency Inversion - Model configuration injected via constructor
    - Open for Extension - Easily extended for domain-specific verification
    
    Verification represents a crucial pattern for mission-critical applications where:
    - Answer correctness is paramount (e.g., medical, legal, or financial domains)
    - Independent quality control is required for regulatory compliance
    - Explainable AI principles must be followed with reasoning transparency
    - Automatic error correction capabilities provide fault tolerance
    
    Example:
        verifier = Verifier(model_name="anthropic:claude-3-opus")
        result = verifier(inputs=VerifierInputs(
            query="What is the boiling point of water?", 
            candidate_answer="Water boils at 90 degrees Celsius at sea level."
        ))
        
        verdict = result.verdict           # "incorrect"
        explanation = result.explanation   # Detailed explanation of the error
        revised = result.revised_answer    # "Water boils at 100 degrees Celsius at sea level."
    """

    specification: Specification = VerifierSpecification()
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


class VariedEnsembleSpecification(Specification):
    input_model: Type[BaseModel] = VariedEnsembleInputs
    output_model: Type[BaseModel] = VariedEnsembleOutputs


class VariedEnsemble(Operator[VariedEnsembleInputs, VariedEnsembleOutputs]):
    """Wrapper around multiple LM modules that runs varied configurations and aggregates outputs.

    Example:
        varied_ensemble = VariedEnsemble(model_configs=[config1, config2])
        outputs = varied_ensemble(inputs=VariedEnsembleInputs(query="Example query"))
    """

    specification: Specification = VariedEnsembleSpecification()
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

        If a prompt_template is defined in the specification, it is used; otherwise, defaults to the query.
        """
        if self.specification and self.specification.prompt_template:
            return self.specification.render_prompt(inputs=inputs)
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
    """
    Chains multiple operators together, passing outputs from one to the next.
    
    This operator executes a sequence of operators in order, where each operator's
    output becomes the input to the next operator. The result is a single combined
    operator that can be used wherever any individual operator is expected.
    
    Example:
        pipeline = Sequential(operators=[
            UniformEnsemble(num_units=3, model_name="gpt-4o"),
            JudgeSynthesis(model_name="claude-3-opus"),
            Verifier(model_name="gpt-4o")
        ])
        
        result = pipeline(inputs={"query": "What causes climate change?"})
    """

    operators: List[Operator[Any, Any]]
    specification: Specification = Specification(input_model=None, output_model=None)

    def __init__(self, *, operators: List[Operator[Any, Any]]) -> None:
        self.operators = operators

    def forward(self, *, inputs: T_in) -> T_out:
        """Executes the operators sequentially."""
        for op in self.operators:
            inputs = op(inputs=inputs)
        return inputs
