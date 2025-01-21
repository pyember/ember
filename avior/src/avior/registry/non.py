# non.py
# ------------------------------------------------------------------------------
# Extended "non" module providing typed wrapper classes around all built-in
# Avior operators from the operator_registry. This includes:
#   1) EnsembleOperator    -> Ensemble
#   2) MostCommonOperator  -> MostCommon
#   3) GetAnswerOperator   -> GetAnswer
#   4) JudgeSynthesisOperator -> JudgeSynthesis
#   5) VerifierOperator    -> Verifier
# ------------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

# Avior imports (update paths if needed):
from src.avior.registry.operator.operator_base import (
    Operator,
    OperatorMetadata,
    OperatorType,
)
from src.avior.registry.prompt_signature.signatures import Signature
# Import the "raw" operators from the registry
from src.avior.registry.operator.operator_registry import (
    EnsembleOperator,
    MostCommonOperator,
    GetAnswerOperator,
    JudgeSynthesisOperator,
    VerifierOperator,
)
from src.avior.registry.model.services.model_service import ModelService
from src.avior.modules.lm_modules import LMModuleConfig, LMModule


# ------------------------------------------------------------------------------
# 1) Ensemble
# ------------------------------------------------------------------------------

class EnsembleInputs(BaseModel):
    """Typed inputs for our Ensemble wrapper."""
    query: str

class EnsembleSignature(Signature):
    required_inputs: List[str] = ["query"]
    input_model: Type[BaseModel] = EnsembleInputs

class Ensemble(Operator[EnsembleInputs, Dict[str, Any]]):
    """
    A wrapper that internally uses EnsembleOperator from the registry to
    do multiple parallel LM calls. We add 'model_service=None' to LMModule
    to address the updated constructor that requires model_service.
    
    Example usage:
        ensemble = Ensemble(num_units=2, model_name="gpt-4-turbo")
        output = ensemble({"query": "What is the capital of France?"})
        # output => {"responses": ["Paris", "Paris", ...]} 
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="ENSEMBLE_WRAPPER",
        description="Wrapper around EnsembleOperator for parallel model calls",
        operator_type=OperatorType.FAN_OUT,
        signature=EnsembleSignature(),
    )

    def __init__(
        self,
        num_units: int = 3,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs
    ):
        """
        :param num_units: Number of LMModules in the ensemble.
        :param model_name: Model name for each LM module.
        :param temperature: Sampling temperature for each LM.
        :param max_tokens: Optional max token limit.
        :param model_service: If None, we create a fresh ModelService via get_default_model_service().
        :param kwargs: Additional keyword args passed to EnsembleOperator.
        """
        super().__init__(
            name="Ensemble",
            signature=self.metadata.signature,
        )
        lm_modules = [
            LMModule(
                config=LMModuleConfig(
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                model_service=model_service
            )
            for _ in range(num_units)
        ]
        self._ensemble_op = EnsembleOperator(lm_modules=lm_modules, **kwargs)
        self.ensemble_op = self._ensemble_op  # so sub-operators auto-discovery sees it

    def forward(self, inputs: EnsembleInputs) -> Dict[str, Any]:
        # Just call the underlying operator with pydantic => dict conversion
        return self._ensemble_op(inputs.model_dump())


# ------------------------------------------------------------------------------
# 2) MostCommon
# ------------------------------------------------------------------------------

class MostCommonInputs(BaseModel):
    """Typed inputs for our MostCommon wrapper."""
    query: str
    responses: List[str]

class MostCommonSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = MostCommonInputs

class MostCommon(Operator[MostCommonInputs, Dict[str, Any]]):
    """
    A wrapper around MostCommonOperator from the registry, which picks the
    single most frequent answer from a list of responses.
    
    Example usage:
        aggregator = MostCommon()
        output = aggregator({"query": "...", "responses": ["A", "B", "A"]})
        # output => {"final_answer": "A"}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="MOST_COMMON_WRAPPER",
        description="Wrapper around MostCommonOperator to pick the best consensus answer",
        operator_type=OperatorType.FAN_IN,
        signature=MostCommonSignature(),
    )

    def __init__(self, **kwargs):
        """
        :param kwargs: Additional arguments to pass to the underlying operator, if any.
        """
        super().__init__(
            name="MostCommon",
            signature=self.metadata.signature,
        )
        self._mc_op = MostCommonOperator(lm_modules=[], **kwargs)
        self.mc_op = self._mc_op

    def forward(self, inputs: MostCommonInputs) -> Dict[str, Any]:
        return self._mc_op(inputs.model_dump())


# ------------------------------------------------------------------------------
# 3) GetAnswer
# ------------------------------------------------------------------------------

class GetAnswerInputs(BaseModel):
    """Typed inputs for GetAnswer wrapper."""
    query: str
    responses: List[str] = Field(..., description="List of response strings to parse into a single final answer.")

class GetAnswerSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = GetAnswerInputs

class GetAnswer(Operator[GetAnswerInputs, Dict[str, Any]]):
    """
    A wrapper around GetAnswerOperator, which typically takes a list of responses,
    possibly uses an LM to parse or extract a final single answer, returning
    {"final_answer": "..."}.
    
    Example usage:
        getter = GetAnswer(model_name="gpt-4o")
        output = getter({"query": "Which label is correct?", "responses": ["A", "B"]})
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="GET_ANSWER_WRAPPER",
        description="Wrapper around GetAnswerOperator, extracting a single answer from multiple responses.",
        operator_type=OperatorType.RECURRENT,
        signature=GetAnswerSignature(),
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs
    ):
        """
        :param model_name: Model name for each LM module.
        :param temperature: Sampling temperature for each LM.
        :param max_tokens: Optional max token limit.
        :param model_service: If None, we create a fresh ModelService via get_default_model_service().
        :param kwargs: Additional keyword args passed to GetAnswerOperator.
        """
        super().__init__(
            name="GetAnswer",
            signature=self.metadata.signature,
        )
        lm_module = LMModule(
            LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service
        )
        self._get_answer_op = GetAnswerOperator(lm_modules=[lm_module], **kwargs)
        self.get_answer_op = self._get_answer_op

    def forward(self, inputs: GetAnswerInputs) -> Dict[str, Any]:
        return self._get_answer_op(inputs.model_dump())


# ------------------------------------------------------------------------------
# 4) JudgeSynthesis
# ------------------------------------------------------------------------------

class JudgeSynthesisInputs(BaseModel):
    """Typed inputs for the JudgeSynthesis wrapper."""
    query: str
    responses: List[str] = Field(..., description="List of responses to synthesize a single best final answer.")

class JudgeSynthesisSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = JudgeSynthesisInputs

class JudgeSynthesis(Operator[JudgeSynthesisInputs, Dict[str, Any]]):
    """
    A wrapper for the JudgeSynthesisOperator, which merges multiple advisor 
    responses into one final, reasoned answer, with optional concurrency.

    Example usage:
        judge = JudgeSynthesis(model_name="gpt-4o")
        out = judge({"query": "What is 2+2?", "responses": ["3","4","2"]})
        # out => {"final_answer": "...", "reasoning": "..."}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="JUDGE_SYNTHESIS_WRAPPER",
        description="Wrapper around JudgeSynthesisOperator for multi-response reasoning.",
        operator_type=OperatorType.FAN_IN,
        signature=JudgeSynthesisSignature(),
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs
    ):
        """
        :param model_name: Model name for each LM module.
        :param temperature: Sampling temperature for each LM.
        :param max_tokens: Optional max token limit.
        :param model_service: If None, we create a fresh ModelService via get_default_model_service().
        :param kwargs: Additional keyword args passed to JudgeSynthesisOperator.
        """
        super().__init__(
            name="JudgeSynthesis",
            signature=self.metadata.signature,
        )
        lm_module = LMModule(
            LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service
        )
        self._judge_synth_op = JudgeSynthesisOperator(lm_modules=[lm_module])
        self.judge_synth_op = self._judge_synth_op

    def forward(self, inputs: JudgeSynthesisInputs) -> Dict[str, Any]:
        return self._judge_synth_op(inputs.model_dump())


# ------------------------------------------------------------------------------
# 5) Verifier
# ------------------------------------------------------------------------------

class VerifierInputs(BaseModel):
    """Typed inputs for the Verifier wrapper."""
    query: str
    candidate_answer: str = Field(..., description="The answer to verify correctness for.")

class VerifierSignature(Signature):
    required_inputs: List[str] = ["query", "candidate_answer"]
    input_model: Type[BaseModel] = VerifierInputs

class Verifier(Operator[VerifierInputs, Dict[str, Any]]):
    """
    A wrapper around VerifierOperator, which checks correctness of a candidate answer,
    optionally revising it if found incorrect.

    Example usage:
        verifier = Verifier(model_name="gpt-4o")
        out = verifier({"query": "What is 2+2?", "candidate_answer": "5"})
        # out => {"verdict":"Incorrect","explanation":"...","revised_answer":"4"}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VERIFIER_WRAPPER",
        description="Wrapper around VerifierOperator to check correctness of a final answer.",
        operator_type=OperatorType.RECURRENT,
        signature=VerifierSignature(),
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs
    ):
        """
        :param model_name: Model name for each LM module.
        :param temperature: Sampling temperature for each LM.
        :param max_tokens: Optional max token limit.
        :param model_service: If None, we create a fresh ModelService via get_default_model_service().
        :param kwargs: Additional keyword args passed to VerifierOperator.
        """
        super().__init__(
            name="Verifier",
            signature=self.metadata.signature,
        )
        lm_module = LMModule(
            LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service
        )
        self._verifier_op = VerifierOperator(lm_modules=[lm_module], **kwargs)
        self.verifier_op = self._verifier_op

    def forward(self, inputs: VerifierInputs) -> Dict[str, Any]:
        return self._verifier_op(inputs.model_dump())

class VariedEnsembleInputs(BaseModel):
    """Inputs for the VariedEnsemble operator."""
    query: str

class VariedEnsembleOutputs(BaseModel):
    """Outputs for the VariedEnsemble operator."""
    responses: List[str]

class VariedEnsembleSignature(Signature):
    required_inputs: List[str] = ["query"]
    input_model: Type[BaseModel] = VariedEnsembleInputs
    # Optionally, you could also specify an output_model if desired

class VariedEnsemble(Operator[VariedEnsembleInputs, VariedEnsembleOutputs]):
    """
    Operator that runs multiple different model configurations in parallel (or sequentially),
    returning the list of responses from each distinct model.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VARIED_ENSEMBLE",
        description="Runs multiple different models in parallel, returning all responses.",
        operator_type=OperatorType.FAN_OUT,
        signature=VariedEnsembleSignature(),
    )

    def __init__(
        self,
        model_configs: List[LMModuleConfig],
        name: str = "VariedEnsemble",
    ):
        """
        :param model_configs: A list of LMModuleConfig objects (one per distinct model).
        :param name: Operator name for debugging / introspection.
        """
        super().__init__(name=name, signature=self.metadata.signature)
        # Create an LMModule for each provided config
        self.lm_modules = [LMModule(config=c) for c in model_configs]

    def forward(self, inputs: VariedEnsembleInputs) -> VariedEnsembleOutputs:
        # Build prompt from the input model (though you can also just use inputs.query directly)
        prompt = self.build_prompt(inputs.model_dump())
        responses: List[str] = []

        # Call each LM, collecting their outputs
        for lm in self.lm_modules:
            resp = self.call_lm(prompt, lm).strip()
            responses.append(resp)

        return VariedEnsembleOutputs(responses=responses)


# ------------------------------------------------------------------------------
# 6) Bringing it All Together
# ------------------------------------------------------------------------------
# 
# This file defines typed wrappers around Avior's built-in operators, each
# subclassing the base Operator and passing typed inputs to the underlying
# registry operator. You can create complex pipelines by composing these 
# wrappers as sub-operators of your custom operator classes (similar to 
# the "NestedNetwork" pattern).
#
# For example, if you want to create an operator that does:
#   Ensemble -> MostCommon -> Verifier
# you can do:
#
#   class MyPipeline(Operator[MyPipelineInputs, Dict[str, Any]]):
#       def __init__(self):
#           super().__init__()
#           self.ensemble = Ensemble(num_units=3, ...)
#           self.most_common = MostCommon()
#           self.verifier = Verifier(...)
#
#       def forward(self, inputs: MyPipelineInputs) -> Dict[str, Any]:
#           out1 = self.ensemble({"query": inputs.query})
#           out2 = self.most_common({"query": inputs.query, "responses": out1["responses"]})
#           out3 = self.verifier({"query": inputs.query, "candidate_answer": out2["final_answer"]})
#           return out3
#
# That's the core pattern for building advanced multi-step pipelines in Avior.
