# ------------------------------------------------------------------------------
# Extended "non" module providing strongly-typed wrappers around built-in
# ember operators from the operator registry. Each wrapper adheres to the
# Google Python Style Guide, leveraging strong type annotations and explicit
# named method invocations for clarity and maintainability.
# ------------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

# ember imports (update paths if needed):
from ember.core.registry.operator.core.operator_base import (
    Operator,
    OperatorMetadata,
)
from ember.core.registry.prompt_signature.signatures import Signature

# Import the "raw" operators from the registry.
from ember.core.registry.operator.operator_registry import (
    EnsembleOperator,
    MostCommonOperator,
    GetAnswerOperator,
    JudgeSynthesisOperator,
    VerifierOperator,
)
from ember.core.registry.model.core.services.model_service import ModelService
from ember.core.registry.model.core.modules.lm_modules import LMModuleConfig, LMModule


# ------------------------------------------------------------------------------
# 1) Ensemble
# ------------------------------------------------------------------------------

class EnsembleInputs(BaseModel):
    """Typed input for the Ensemble operator.

    Attributes:
        query (str): The query string to be processed.
    """
    query: str


class EnsembleSignature(Signature):
    required_inputs: List[str] = ["query"]
    input_model: Type[BaseModel] = EnsembleInputs


class Ensemble(Operator[EnsembleInputs, Dict[str, Any]]):
    """Wrapper around EnsembleOperator for parallel model calls.

    This operator instantiates multiple LMModules and leverages the underlying
    EnsembleOperator to execute parallel language model calls.

    Example:
        ensemble = Ensemble(num_units=2, model_name="gpt-4-turbo")
        output = ensemble({"query": "What is the capital of France?"})
        # output: {"responses": ["Paris", "Paris", ...]}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="ENSEMBLE_WRAPPER",
        description="Wrapper around EnsembleOperator for parallel model calls",
        signature=EnsembleSignature(),
    )

    def __init__(
        self,
        num_units: int = 3,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Ensemble operator.

        Args:
            num_units (int): Number of LMModules in the ensemble.
            model_name (str): Model name for each LMModule.
            temperature (float): Sampling temperature for LM calls.
            max_tokens (Optional[int]): Optional maximum token limit.
            model_service (Optional[ModelService]): Model service for LMModules; if None, a
                default service is used.
            **kwargs: Additional keyword arguments forwarded to EnsembleOperator.
        """
        super().__init__(name="Ensemble", signature=self.metadata.signature)
        lm_modules: List[LMModule] = [
            LMModule(
                config=LMModuleConfig(
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                model_service=model_service,
            )
            for _ in range(num_units)
        ]
        self._ensemble_op: EnsembleOperator = EnsembleOperator(lm_modules=lm_modules, **kwargs)
        self.ensemble_op: EnsembleOperator = self._ensemble_op  # For sub-operator auto-discovery.

    def forward(self, inputs: EnsembleInputs) -> Dict[str, Any]:
        """Executes the ensemble operation with the provided inputs.

        Args:
            inputs (EnsembleInputs): The input parameters containing the query.

        Returns:
            Dict[str, Any]: A dictionary with the responses from each LMModule.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._ensemble_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 2) MostCommon
# ------------------------------------------------------------------------------

class MostCommonInputs(BaseModel):
    """Typed input for the MostCommon operator.

    Attributes:
        query (str): The initial query.
        responses (List[str]): Candidate responses from which the most common answer is determined.
    """
    query: str
    responses: List[str]


class MostCommonSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = MostCommonInputs


class MostCommon(Operator[MostCommonInputs, Dict[str, Any]]):
    """Wrapper around MostCommonOperator to determine the consensus answer.

    Example:
        aggregator = MostCommon()
        output = aggregator({"query": "...", "responses": ["A", "B", "A"]})
        # output: {"final_answer": "A"}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="MOST_COMMON_WRAPPER",
        description="Wrapper around MostCommonOperator to pick the best consensus answer",
        signature=MostCommonSignature(),
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the MostCommon operator.

        Args:
            **kwargs: Additional arguments to pass to MostCommonOperator.
        """
        super().__init__(name="MostCommon", signature=self.metadata.signature)
        self._mc_op: MostCommonOperator = MostCommonOperator(lm_modules=[], **kwargs)
        self.mc_op: MostCommonOperator = self._mc_op

    def forward(self, inputs: MostCommonInputs) -> Dict[str, Any]:
        """Processes inputs to determine the most common response.

        Args:
            inputs (MostCommonInputs): The input parameters including query and responses.

        Returns:
            Dict[str, Any]: A dictionary containing the final consensus answer.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._mc_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 3) GetAnswer
# ------------------------------------------------------------------------------

class GetAnswerInputs(BaseModel):
    """Typed input for the GetAnswer operator.

    Attributes:
        query (str): The query for which an answer is being sought.
        responses (List[str]): A list of candidate response strings.
    """
    query: str
    responses: List[str] = Field(
        ...,
        description="List of response strings to parse into a single final answer."
    )


class GetAnswerSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = GetAnswerInputs


class GetAnswer(Operator[GetAnswerInputs, Dict[str, Any]]):
    """Wrapper around GetAnswerOperator to extract a single answer from multiple responses.

    Example:
        getter = GetAnswer(model_name="gpt-4o")
        output = getter({"query": "Which label is correct?", "responses": ["A", "B"]})
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="GET_ANSWER_WRAPPER",
        description="Wrapper around GetAnswerOperator, extracting a single answer from multiple responses.",
        signature=GetAnswerSignature(),
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the GetAnswer operator.

        Args:
            model_name (str): Model name for the LMModule.
            temperature (float): Sampling temperature for LM calls.
            max_tokens (Optional[int]): Optional maximum token limit.
            model_service (Optional[ModelService]): Model service for LMModule.
            **kwargs: Additional keyword arguments for GetAnswerOperator.
        """
        super().__init__(name="GetAnswer", signature=self.metadata.signature)
        lm_module: LMModule = LMModule(
            LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service,
        )
        self._get_answer_op: GetAnswerOperator = GetAnswerOperator(lm_modules=[lm_module], **kwargs)
        self.get_answer_op: GetAnswerOperator = self._get_answer_op

    def forward(self, inputs: GetAnswerInputs) -> Dict[str, Any]:
        """Extracts a final answer from multiple candidate responses.

        Args:
            inputs (GetAnswerInputs): The input parameters including query and candidate responses.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted final answer.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._get_answer_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 4) JudgeSynthesis
# ------------------------------------------------------------------------------

class JudgeSynthesisInputs(BaseModel):
    """Typed input for the JudgeSynthesis operator.

    Attributes:
        query (str): The query for synthesis.
        responses (List[str]): Responses to combine into a final answer.
    """
    query: str
    responses: List[str] = Field(
        ...,
        description="List of responses to synthesize a single best final answer."
    )


class JudgeSynthesisSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = JudgeSynthesisInputs


class JudgeSynthesis(Operator[JudgeSynthesisInputs, Dict[str, Any]]):
    """Wrapper around JudgeSynthesisOperator for multi-response reasoning.

    This operator fuses multiple advisor responses into a final, reasoned answer,
    optionally with concurrent execution.

    Example:
        judge = JudgeSynthesis(model_name="gpt-4o")
        output = judge({"query": "What is 2+2?", "responses": ["3", "4", "2"]})
        # output: {"final_answer": "...", "reasoning": "..."}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="JUDGE_SYNTHESIS_WRAPPER",
        description="Wrapper around JudgeSynthesisOperator for multi-response reasoning.",
        signature=JudgeSynthesisSignature(),
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the JudgeSynthesis operator.

        Args:
            model_name (str): Model name for the LMModule.
            temperature (float): Sampling temperature for LM calls.
            max_tokens (Optional[int]): Optional maximum token limit.
            model_service (Optional[ModelService]): Model service for LMModule.
            **kwargs: Additional keyword arguments for JudgeSynthesisOperator.
        """
        super().__init__(name="JudgeSynthesis", signature=self.metadata.signature)
        lm_module: LMModule = LMModule(
            LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service,
        )
        self._judge_synth_op: JudgeSynthesisOperator = JudgeSynthesisOperator(lm_modules=[lm_module])
        self.judge_synth_op: JudgeSynthesisOperator = self._judge_synth_op

    def forward(self, inputs: JudgeSynthesisInputs) -> Dict[str, Any]:
        """Synthesizes a final answer from multiple responses.

        Args:
            inputs (JudgeSynthesisInputs): The input parameters including query and responses.

        Returns:
            Dict[str, Any]: A dictionary containing the final answer and supporting reasoning.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._judge_synth_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 5) Verifier
# ------------------------------------------------------------------------------

class VerifierInputs(BaseModel):
    """Typed input for the Verifier operator.

    Attributes:
        query (str): The query string.
        candidate_answer (str): The answer for which correctness is to be verified.
    """
    query: str
    candidate_answer: str = Field(
        ...,
        description="The answer to verify correctness for."
    )


class VerifierSignature(Signature):
    required_inputs: List[str] = ["query", "candidate_answer"]
    input_model: Type[BaseModel] = VerifierInputs


class Verifier(Operator[VerifierInputs, Dict[str, Any]]):
    """Wrapper around VerifierOperator to evaluate and potentially revise a candidate answer.

    Example:
        verifier = Verifier(model_name="gpt-4o")
        output = verifier({"query": "What is 2+2?", "candidate_answer": "5"})
        # output: {"verdict": "Incorrect", "explanation": "...", "revised_answer": "4"}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VERIFIER_WRAPPER",
        description="Wrapper around VerifierOperator to check correctness of a final answer.",
        signature=VerifierSignature(),
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Verifier operator.

        Args:
            model_name (str): Model name for the LMModule.
            temperature (float): Sampling temperature for LM calls.
            max_tokens (Optional[int]): Optional maximum token limit.
            model_service (Optional[ModelService]): Model service for LMModule.
            **kwargs: Additional keyword arguments for VerifierOperator.
        """
        super().__init__(name="Verifier", signature=self.metadata.signature)
        lm_module: LMModule = LMModule(
            LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service,
        )
        self._verifier_op: VerifierOperator = VerifierOperator(lm_modules=[lm_module], **kwargs)
        self.verifier_op: VerifierOperator = self._verifier_op

    def forward(self, inputs: VerifierInputs) -> Dict[str, Any]:
        """Verifies the correctness of a candidate answer.

        Args:
            inputs (VerifierInputs): The input parameters including query and candidate answer.

        Returns:
            Dict[str, Any]: A dictionary containing the verification verdict, explanation,
            and an optional revised answer.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._verifier_op.forward(inputs=input_data)


class VariedEnsembleInputs(BaseModel):
    """Typed input for the VariedEnsemble operator.

    Attributes:
        query (str): The query to be processed across various model configurations.
    """
    query: str


class VariedEnsembleOutputs(BaseModel):
    """Typed output for the VariedEnsemble operator.

    Attributes:
        responses (List[str]): A list of responses from the different LM configurations.
    """
    responses: List[str]


class VariedEnsembleSignature(Signature):
    required_inputs: List[str] = ["query"]
    input_model: Type[BaseModel] = VariedEnsembleInputs
    # Optionally, an output_model can be specified if desired.


class VariedEnsemble(Operator[VariedEnsembleInputs, VariedEnsembleOutputs]):
    """Operator that executes multiple LM configurations in parallel (or sequentially)
    and aggregates their responses.

    Example:
        varied_ensemble = VariedEnsemble(model_configs=[config1, config2])
        outputs = varied_ensemble({"query": "Example query"})
        # outputs: VariedEnsembleOutputs with responses from all models.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VARIED_ENSEMBLE",
        description="Runs multiple different models in parallel, returning all responses.",
        signature=VariedEnsembleSignature(),
    )

    def __init__(
        self,
        model_configs: List[LMModuleConfig],
        name: str = "VariedEnsemble",
    ) -> None:
        """Initializes the VariedEnsemble operator.

        Args:
            model_configs (List[LMModuleConfig]): A list of LMModuleConfig objects, one per distinct model.
            name (str): Operator name for debugging and introspection.
        """
        super().__init__(name=name, signature=self.metadata.signature)
        self.lm_modules: List[LMModule] = [LMModule(config=c) for c in model_configs]

    def forward(self, inputs: VariedEnsembleInputs) -> VariedEnsembleOutputs:
        """Executes the varied ensemble operation and returns the aggregated responses.

        Args:
            inputs (VariedEnsembleInputs): The input parameters containing the query.

        Returns:
            VariedEnsembleOutputs: An output object containing responses from each LM configuration.
        """
        input_dict: Dict[str, Any] = inputs.model_dump()
        prompt: str = self.build_prompt(inputs=input_dict)
        responses: List[str] = []

        # Call each LM, collecting their outputs
        for lm in self.lm_modules:
            response_text: str = self.call_lm(prompt=prompt, lm=lm).strip()
            responses.append(response_text)

        return VariedEnsembleOutputs(responses=responses)


# ------------------------------------------------------------------------------
# 6) Bringing it All Together
# ------------------------------------------------------------------------------
#
# This module defines strongly-typed wrappers around ember's built-in operators.
# Each wrapper subclasses the base Operator and enforces typed inputs on the
# underlying registry operator. By composing these wrappers, complex multi-step
# pipelines can be built—following the core "NestedNetwork" pattern.
#
# For example, to compose a pipeline of Ensemble -> MostCommon -> Verifier:
#
#     class MyPipeline(Operator[MyPipelineInputs, Dict[str, Any]]):
#         def __init__(self) -> None:
#             super().__init__()
#             self.ensemble = Ensemble(num_units=3, ...)
#             self.most_common = MostCommon()
#             self.verifier = Verifier(...)
#
#       def forward(self, inputs: MyPipelineInputs) -> Dict[str, Any]:
#           out1 = self.ensemble({"query": inputs.query})
#           out2 = self.most_common({"query": inputs.query, "responses": out1["responses"]})
#           out3 = self.verifier({"query": inputs.query, "candidate_answer": out2["final_answer"]})
#           return out3
#
# This pattern serves as the foundation for building advanced, multi-step pipelines in ember.
