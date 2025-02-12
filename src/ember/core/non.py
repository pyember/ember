# ------------------------------------------------------------------------------
# Extended "non" module providing strongly typed wrappers for the built-in
# ember operators from the operator registry. Each wrapper adheres to the
# Google Python Style Guide with comprehensive type annotations and explicit
# named method invocations, ensuring clarity, maintainability, and scalability.
# ------------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

# Ember package imports:
from ember.core.registry.operator.core.operator_base import (
    Operator,
    OperatorMetadata,
    T_in,
    T_out,
)
from ember.core.registry.prompt_signature.signatures import Signature
from ember.core.registry.operator.operator_registry import (
    EnsembleOperator,
    MostCommonOperator,
    GetAnswerOperator,
    JudgeSynthesisOperator,
    VerifierOperator,
)
from ember.core.registry.model.services.model_service import ModelService
from ember.core.registry.model.modules.lm import LMModuleConfig, LMModule


# ------------------------------------------------------------------------------
# 1) Ensemble Operator Wrapper
# ------------------------------------------------------------------------------


class EnsembleInputs(BaseModel):
    """Represents the typed input for the Ensemble operator.

    Attributes:
        query (str): The input query to be processed.
    """

    query: str


class EnsembleSignature(Signature):
    required_inputs: List[str] = ["query"]
    input_model: Type[BaseModel] = EnsembleInputs


class Ensemble(Operator[EnsembleInputs, Dict[str, Any]]):
    """Wrapper around EnsembleOperator for executing parallel LM module calls.

    This operator creates multiple LMModule instances and leverages the underlying
    EnsembleOperator to perform parallel language model invocations.

    Example:
        ensemble = Ensemble(num_units=2, model_name="gpt-4-turbo")
        output = ensemble(inputs=EnsembleInputs(query="What is the capital of France?"))
        # output: {"responses": ["Paris", "Paris", ...]}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="ENSEMBLE_WRAPPER",
        description="Wrapper around EnsembleOperator for parallel model calls.",
        signature=EnsembleSignature(),
    )

    def __init__(
        self,
        *,
        num_units: int = 3,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Ensemble operator.

        Args:
            num_units (int): Number of LMModule instances to instantiate.
            model_name (str): Identifier for the model used in each LMModule.
            temperature (float): Temperature parameter for LM sampling.
            max_tokens (Optional[int]): Maximum tokens allowed per LM call.
            model_service (Optional[ModelService]): Service instance for LMModules; if None, defaults are used.
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
        self._ensemble_op: EnsembleOperator = EnsembleOperator(
            lm_modules=lm_modules, **kwargs
        )
        self.ensemble_op: EnsembleOperator = (
            self._ensemble_op
        )  # For sub-operator auto-discovery.

    def forward(self, *, inputs: EnsembleInputs) -> Dict[str, Any]:
        """Executes the ensemble operation on the provided inputs.

        Args:
            inputs (EnsembleInputs): Input parameters containing the query.

        Returns:
            Dict[str, Any]: Dictionary with responses from each LMModule.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._ensemble_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 2) MostCommon Operator Wrapper
# ------------------------------------------------------------------------------


class MostCommonInputs(BaseModel):
    """Represents the typed input for the MostCommon operator.

    Attributes:
        query (str): The initial query.
        responses (List[str]): List of candidate responses to determine consensus.
    """

    query: str
    responses: List[str]


class MostCommonSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = MostCommonInputs


class MostCommon(Operator[MostCommonInputs, Dict[str, Any]]):
    """Wrapper around MostCommonOperator for determining a consensus answer.

    Example:
        aggregator = MostCommon()
        output = aggregator(inputs=MostCommonInputs(query="...", responses=["A", "B", "A"]))
        # output: {"final_answer": "A"}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="MOST_COMMON_WRAPPER",
        description="Wrapper around MostCommonOperator to select the best consensus answer.",
        signature=MostCommonSignature(),
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the MostCommon operator.

        Args:
            **kwargs: Additional arguments forwarded to MostCommonOperator.
        """
        super().__init__(name="MostCommon", signature=self.metadata.signature)
        self._mc_op: MostCommonOperator = MostCommonOperator(lm_modules=[], **kwargs)
        self.mc_op: MostCommonOperator = self._mc_op

    def forward(self, *, inputs: MostCommonInputs) -> Dict[str, Any]:
        """Determines the most common response from candidate responses.

        Args:
            inputs (MostCommonInputs): Input parameters including query and responses.

        Returns:
            Dict[str, Any]: Dictionary with the final consensus answer.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._mc_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 3) GetAnswer Operator Wrapper
# ------------------------------------------------------------------------------


class GetAnswerInputs(BaseModel):
    """Represents the typed input for the GetAnswer operator.

    Attributes:
        query (str): The query for which an answer is sought.
        responses (List[str]): List of candidate responses to be processed.
    """

    query: str
    responses: List[str] = Field(
        ...,
        description="List of candidate response strings from which a final answer is extracted.",
    )


class GetAnswerSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = GetAnswerInputs


class GetAnswer(Operator[GetAnswerInputs, Dict[str, Any]]):
    """Wrapper around GetAnswerOperator to extract a single final answer.

    Example:
        getter = GetAnswer(model_name="gpt-4o")
        output = getter(inputs=GetAnswerInputs(query="Which label is correct?", responses=["A", "B"]))
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="GET_ANSWER_WRAPPER",
        description="Wrapper around GetAnswerOperator to extract a single answer from multiple responses.",
        signature=GetAnswerSignature(),
    )

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the GetAnswer operator.

        Args:
            model_name (str): Identifier for the LMModule model.
            temperature (float): Temperature parameter for LM sampling.
            max_tokens (Optional[int]): Maximum tokens allowed per LM call.
            model_service (Optional[ModelService]): Service instance for LMModule; defaults if None.
            **kwargs: Additional keyword arguments forwarded to GetAnswerOperator.
        """
        super().__init__(name="GetAnswer", signature=self.metadata.signature)
        lm_module: LMModule = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service,
        )
        self._get_answer_op: GetAnswerOperator = GetAnswerOperator(
            lm_modules=[lm_module], **kwargs
        )
        self.get_answer_op: GetAnswerOperator = self._get_answer_op

    def forward(self, *, inputs: GetAnswerInputs) -> Dict[str, Any]:
        """Extracts a single final answer from multiple candidate responses.

        Args:
            inputs (GetAnswerInputs): Input parameters including query and candidate responses.

        Returns:
            Dict[str, Any]: Dictionary containing the extracted final answer.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._get_answer_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 4) JudgeSynthesis Operator Wrapper
# ------------------------------------------------------------------------------


class JudgeSynthesisInputs(BaseModel):
    """Represents the typed input for the JudgeSynthesis operator.

    Attributes:
        query (str): The query to be synthesized.
        responses (List[str]): List of responses to be aggregated into a final answer.
    """

    query: str
    responses: List[str] = Field(
        ...,
        description="List of responses to synthesize into a single, reasoned answer.",
    )


class JudgeSynthesisSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    input_model: Type[BaseModel] = JudgeSynthesisInputs


class JudgeSynthesis(Operator[JudgeSynthesisInputs, Dict[str, Any]]):
    """Wrapper around JudgeSynthesisOperator for multi-response reasoning synthesis.

    Example:
        judge = JudgeSynthesis(model_name="gpt-4o")
        output = judge(inputs=JudgeSynthesisInputs(query="What is 2+2?", responses=["3", "4", "2"]))
        # output: {"final_answer": "...", "reasoning": "..."}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="JUDGE_SYNTHESIS_WRAPPER",
        description="Wrapper around JudgeSynthesisOperator for synthesizing multi-response answers.",
        signature=JudgeSynthesisSignature(),
    )

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the JudgeSynthesis operator.

        Args:
            model_name (str): Identifier for the LMModule model.
            temperature (float): Temperature parameter for LM sampling.
            max_tokens (Optional[int]): Maximum tokens allowed per LM call.
            model_service (Optional[ModelService]): Service instance for LMModule; using default if None.
            **kwargs: Additional keyword arguments forwarded to JudgeSynthesisOperator.
        """
        super().__init__(name="JudgeSynthesis", signature=self.metadata.signature)
        lm_module: LMModule = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service,
        )
        self._judge_synth_op: JudgeSynthesisOperator = JudgeSynthesisOperator(
            lm_modules=[lm_module]
        )
        self.judge_synth_op: JudgeSynthesisOperator = self._judge_synth_op

    def forward(self, *, inputs: JudgeSynthesisInputs) -> Dict[str, Any]:
        """Synthesizes a final answer from multiple responses.

        Args:
            inputs (JudgeSynthesisInputs): Input parameters including query and responses.

        Returns:
            Dict[str, Any]: Dictionary containing the final answer and reasoned explanation.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._judge_synth_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 5) Verifier Operator Wrapper
# ------------------------------------------------------------------------------


class VerifierInputs(BaseModel):
    """Represents the typed input for the Verifier operator.

    Attributes:
        query (str): The query string.
        candidate_answer (str): The candidate answer to be verified.
    """

    query: str
    candidate_answer: str = Field(
        ...,
        description="The candidate answer whose correctness is to be verified.",
    )


class VerifierSignature(Signature):
    required_inputs: List[str] = ["query", "candidate_answer"]
    input_model: Type[BaseModel] = VerifierInputs


class Verifier(Operator[VerifierInputs, Dict[str, Any]]):
    """Wrapper around VerifierOperator to evaluate and potentially revise a candidate answer.

    Example:
        verifier = Verifier(model_name="gpt-4o")
        output = verifier(inputs=VerifierInputs(query="What is 2+2?", candidate_answer="5"))
        # output: {"verdict": "Incorrect", "explanation": "...", "revised_answer": "4"}
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VERIFIER_WRAPPER",
        description="Wrapper around VerifierOperator for verifying and revising candidate answers.",
        signature=VerifierSignature(),
    )

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        model_service: Optional[ModelService] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Verifier operator.

        Args:
            model_name (str): Identifier for the LMModule model.
            temperature (float): Temperature parameter for LM sampling.
            max_tokens (Optional[int]): Maximum tokens allowed per LM call.
            model_service (Optional[ModelService]): Service instance for LMModule; defaults used if None.
            **kwargs: Additional keyword arguments forwarded to VerifierOperator.
        """
        super().__init__(name="Verifier", signature=self.metadata.signature)
        lm_module: LMModule = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model_service=model_service,
        )
        self._verifier_op: VerifierOperator = VerifierOperator(
            lm_modules=[lm_module], **kwargs
        )
        self.verifier_op: VerifierOperator = self._verifier_op

    def forward(self, *, inputs: VerifierInputs) -> Dict[str, Any]:
        """Verifies the correctness of a candidate answer.

        Args:
            inputs (VerifierInputs): Input parameters including query and candidate answer.

        Returns:
            Dict[str, Any]: Dictionary with verification verdict, explanation, and optional revised answer.
        """
        input_data: Dict[str, Any] = inputs.model_dump()
        return self._verifier_op.forward(inputs=input_data)


# ------------------------------------------------------------------------------
# 6) VariedEnsemble Operator Wrapper and Sequential Pipeline
# ------------------------------------------------------------------------------


class VariedEnsembleInputs(BaseModel):
    """Represents the typed input for the VariedEnsemble operator.

    Attributes:
        query (str): The query to be processed across various model configurations.
    """

    query: str


class VariedEnsembleOutputs(BaseModel):
    """Represents the typed output for the VariedEnsemble operator.

    Attributes:
        responses (List[str]): Collection of responses from different LM configurations.
    """

    responses: List[str]


class VariedEnsembleSignature(Signature):
    required_inputs: List[str] = ["query"]
    input_model: Type[BaseModel] = VariedEnsembleInputs
    # Optionally, an output_model may be defined.


class VariedEnsemble(Operator[VariedEnsembleInputs, VariedEnsembleOutputs]):
    """Executes multiple LM configurations in parallel or sequentially and aggregates their outputs.

    Example:
        varied_ensemble = VariedEnsemble(model_configs=[config1, config2])
        outputs = varied_ensemble(inputs=VariedEnsembleInputs(query="Example query"))
        # outputs: VariedEnsembleOutputs containing responses from each model.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VARIED_ENSEMBLE",
        description="Runs multiple distinct models in parallel, returning all responses.",
        signature=VariedEnsembleSignature(),
    )

    def __init__(
        self,
        *,
        model_configs: List[LMModuleConfig],
        name: str = "VariedEnsemble",
    ) -> None:
        """Initializes the VariedEnsemble operator.

        Args:
            model_configs (List[LMModuleConfig]): A list of LMModuleConfig objects for each distinct model.
            name (str): Operator name for debugging and introspection.
        """
        super().__init__(name=name, signature=self.metadata.signature)
        self.lm_modules: List[LMModule] = [
            LMModule(config=config) for config in model_configs
        ]

    def forward(self, *, inputs: VariedEnsembleInputs) -> VariedEnsembleOutputs:
        """Executes the varied ensemble operation and aggregates the responses.

        Args:
            inputs (VariedEnsembleInputs): Input parameters containing the query.

        Returns:
            VariedEnsembleOutputs: Typed output containing responses from each LM configuration.
        """
        input_dict: Dict[str, Any] = inputs.model_dump()
        prompt: str = self.build_prompt(inputs=input_dict)
        responses: List[str] = []
        # Iterate over each LMModule and collect responses.
        for lm in self.lm_modules:
            response_text: str = self.call_lm(prompt=prompt, lm=lm).strip()
            responses.append(response_text)
        return VariedEnsembleOutputs(responses=responses)


class Sequential(Operator[T_in, T_out]):
    """Compositional operator chaining multiple operators in sequence.

    Example:
        pipeline = Sequential(operators=[
            EnsembleOperator(...),
            JudgeSynthesisOperator(...),
            VerifierOperator(...)
        ])
        final_output = pipeline(inputs={"query": "Hello?"})
    """

    def __init__(self, *, operators: List[Operator[Any, Any]]) -> None:
        """Initializes the Sequential pipeline operator.

        Args:
            operators (List[Operator[Any, Any]]): List of operator instances to be chained.
        """
        super().__init__(name="SequentialPipeline")
        self._operators: List[Operator[Any, Any]] = operators

    def forward(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Executes the chained operators sequentially.

        Args:
            inputs (Union[T_in, Dict[str, Any]]): Initial input data for the pipeline.

        Returns:
            T_out: Final output after all operators have been applied.
        """
        data: Union[T_in, Dict[str, Any]] = inputs
        # Sequentially invoke each operator with named parameters.
        for operator in self._operators:
            data = operator(inputs=data)
        return data  # type: ignore


# ------------------------------------------------------------------------------
# 7) Bringing It All Together
# ------------------------------------------------------------------------------
#
# This module defines strongly typed wrappers for ember's built-in operators.
# Each wrapper subclasses the base Operator and enforces stringent type safety
# for inputs, processing, and outputs of the underlying registry operators.
# By composing these wrappers, robust multi-step pipelines can be built following
# the core "NestedNetwork" pattern.
#
# Example pipeline:
#
#     class MyPipeline(Operator[MyPipelineInputs, Dict[str, Any]]):
#         def __init__(self) -> None:
#             super().__init__()
#             self.ensemble = Ensemble(num_units=3, ...)
#             self.most_common = MostCommon()
#             self.verifier = Verifier(...)
#
#         def forward(self, *, inputs: MyPipelineInputs) -> Dict[str, Any]:
#             out1 = self.ensemble(inputs=EnsembleInputs(query=inputs.query))
#       def forward(self, inputs: MyPipelineInputs) -> Dict[str, Any]:
#           out1 = self.ensemble({"query": inputs.query})
#           out2 = self.most_common({"query": inputs.query, "responses": out1["responses"]})
#           out3 = self.verifier({"query": inputs.query, "candidate_answer": out2["final_answer"]})
#           return out3
#
# This pattern serves as the foundation for building advanced, multi-step pipelines in ember.
#
# This pattern serves as the foundation for building advanced, multi-step pipelines in ember.
