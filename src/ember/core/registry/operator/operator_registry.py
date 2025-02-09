from typing import Any, Dict, List, Optional, Type, TypeVar
from collections import Counter
from pydantic import BaseModel, Field

from ember.core.registry.operator.core.operator_base import (
    Operator,
    OperatorMetadata,
    LMModule,
)
from ember.core.registry.prompt_signature.signatures import Signature
from ember.xcs.scheduler import ExecutionPlan, ExecutionTask

T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out")


class OperatorRegistry:
    """Global registry mapping operator codes to operator classes.

    This registry provides mechanisms to register operator classes with unique codes
    and to retrieve them during runtime.

    Attributes:
        _registry (Dict[str, Type[Operator[Any, Any]]]): Internal mapping from operator codes
            to their corresponding classes.
    """

    def __init__(self) -> None:
        """Initializes an empty OperatorRegistry."""
        self._registry: Dict[str, Type[Operator[Any, Any]]] = {}

    def register(
        self, operator_code: str, operator_cls: Type[Operator[Any, Any]]
    ) -> None:
        """Registers an operator class with a unique code.

        Args:
            operator_code (str): A unique string identifier for the operator.
            operator_cls (Type[Operator[Any, Any]]): The operator class to register.
        """
        self._registry[operator_code] = operator_cls

    def get(self, operator_code: str) -> Optional[Type[Operator[Any, Any]]]:
        """Retrieves an operator class by its unique code.

        Args:
            operator_code (str): The unique identifier for the operator.

        Returns:
            Optional[Type[Operator[Any, Any]]]: The operator class if found; otherwise, None.
        """
        return self._registry.get(operator_code)


####################################################
# Global Registry Instance and Decorator
####################################################

OperatorRegistryGlobal: OperatorRegistry = OperatorRegistry()


def register_operator(registry: OperatorRegistry, code: Optional[str] = None):
    """Decorator to automatically register an operator with the provided registry.

    If no explicit code is given, the decorator uses the operator code specified in the class metadata.

    Args:
        registry (OperatorRegistry): The operator registry instance.
        code (Optional[str]): An explicit operator code override (if provided).

    Returns:
        Callable[[Type[Operator[Any, Any]]], Type[Operator[Any, Any]]]: The decorator function.
    """

    def decorator(cls: Type[Operator[Any, Any]]) -> Type[Operator[Any, Any]]:
        operator_code: str = code if code is not None else cls.metadata.code
        registry.register(operator_code, cls)
        return cls

    return decorator


####################################################
# 1) EnsembleOperator
####################################################


class EnsembleOperatorInputs(BaseModel):
    """Input data model for EnsembleOperator.

    Attributes:
        query (str): The query string used to render the prompt.
    """

    query: str


@register_operator(registry=OperatorRegistryGlobal)
class EnsembleOperator(Operator[EnsembleOperatorInputs, Dict[str, Any]]):
    """Operator that executes parallel calls to multiple LMModules.

    Each LMModule processes the same prompt concurrently and produces its respective response.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="ENSEMBLE",
        description="Runs an ensemble of models to generate responses",
        signature=Signature(input_model=EnsembleOperatorInputs),
    )

    def forward(self, inputs: EnsembleOperatorInputs) -> Dict[str, Any]:
        """Performs a forward pass by concurrently invoking all attached LMModules.

        Args:
            inputs (EnsembleOperatorInputs): The input containing the query string.

        Returns:
            Dict[str, Any]: A dictionary with the key 'responses' mapping to the list of responses.
        """
        prompt: str = self.metadata.signature.render_prompt(inputs=inputs.model_dump())
        responses: List[str] = [lm(prompt=prompt) for lm in self.lm_modules]
        return {"responses": responses}

    def to_plan(self, inputs: EnsembleOperatorInputs) -> Optional[ExecutionPlan]:
        """Creates an execution plan to invoke each LMModule in parallel.

        Args:
            inputs (EnsembleOperatorInputs): The input containing the query.

        Returns:
            Optional[ExecutionPlan]: An execution plan if LMModules exist; otherwise, None.
        """
        if not self.lm_modules:
            return None
        prompt: str = self.metadata.signature.render_prompt(inputs=inputs.model_dump())
        plan: ExecutionPlan = ExecutionPlan()
        for index, lm in enumerate(self.lm_modules):
            task_id: str = f"ensemble_task_{index}"
            plan.add_task(
                ExecutionTask(
                    task_id=task_id,
                    function=self._lm_call_wrapper,
                    inputs={"prompt": prompt, "lm": lm},
                    dependencies=[],
                )
            )
        return plan

    def _lm_call_wrapper(self, *, prompt: str, lm: LMModule) -> str:
        """Wrapper to invoke an LMModule using the provided prompt.

        Args:
            prompt (str): The prompt string to send to the LMModule.
            lm (LMModule): The LMModule instance to be invoked.

        Returns:
            str: The response from the LMModule.
        """
        return lm(prompt=prompt)

    def combine_plan_results(
        self, results: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combines results from an execution plan into a consolidated response.

        Args:
            results (Dict[str, Any]): A mapping from task IDs to their individual results.
            inputs (Dict[str, Any]): The original input dictionary.

        Returns:
            Dict[str, Any]: A dictionary with the key 'responses' mapping to the ordered list of responses.
        """
        sorted_task_ids: List[str] = sorted(results.keys())
        sorted_responses: List[Any] = [results[task_id] for task_id in sorted_task_ids]
        return {"responses": sorted_responses}


####################################################
# 2) MostCommonOperator
####################################################


class MostCommonOperatorInputs(BaseModel):
    """Input data model for MostCommonOperator.

    Attributes:
        query (str): The query string.
        responses (List[str]): A list of response strings.
    """

    query: str
    responses: List[str]


@register_operator(registry=OperatorRegistryGlobal)
class MostCommonOperator(Operator[MostCommonOperatorInputs, Dict[str, Any]]):
    """Operator that selects the most common response from a collection of responses.

    This operator aggregates the responses and determines the most frequently occurring answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="MOST_COMMON",
        description="Determines the most common answer from responses",
        signature=Signature(input_model=MostCommonOperatorInputs),
    )

    def forward(self, inputs: MostCommonOperatorInputs) -> Dict[str, Any]:
        """Selects and returns the most common response.

        Args:
            inputs (MostCommonOperatorInputs): The input containing the query and responses.

        Returns:
            Dict[str, Any]: A dictionary with the key 'final_answer' set to the most common response.
        """
        if not inputs.responses:
            return {"final_answer": None}
        counts: Counter[str] = Counter(inputs.responses)
        final_answer, _ = counts.most_common(1)[0]
        return {"final_answer": final_answer}

    def to_plan(self, inputs: MostCommonOperatorInputs) -> Optional[ExecutionPlan]:
        """Generates an execution plan to concurrently aggregate response frequencies.

        Args:
            inputs (MostCommonOperatorInputs): The input containing the query and responses.

        Returns:
            Optional[ExecutionPlan]: An execution plan if responses exist; otherwise, None.
        """
        if not inputs.responses:
            return None

        plan: ExecutionPlan = ExecutionPlan()
        # Create counting tasks for each individual response.
        for index, response in enumerate(inputs.responses):
            task_id: str = f"count_task_{index}"
            plan.add_task(
                ExecutionTask(
                    task_id=task_id,
                    function=self._count_single_response,
                    inputs={"response": response},
                    dependencies=[],
                )
            )
        # Create an aggregator task to determine the most common response.
        aggregator_task_id: str = "aggregate_most_common"
        plan.add_task(
            ExecutionTask(
                task_id=aggregator_task_id,
                function=self._aggregate_counts,
                inputs={},
                dependencies=[f"count_task_{i}" for i in range(len(inputs.responses))],
            )
        )
        return plan

    def _count_single_response(self, *, response: str) -> Dict[str, int]:
        """Counts the occurrence of a single response.

        Args:
            response (str): A response string.

        Returns:
            Dict[str, int]: A dictionary mapping the response to the count 1.
        """
        return {response: 1}

    def _aggregate_counts(self, **kwargs: Any) -> Dict[str, Any]:
        """Aggregates individual count dictionaries to determine the most common response.

        Args:
            **kwargs (Any): A collection of count dictionaries from individual tasks.

        Returns:
            Dict[str, Any]: A dictionary with the key 'final_answer' set to the most common response.
        """
        aggregate_counter: Counter[str] = Counter()
        for value in kwargs.values():
            if isinstance(value, dict):
                aggregate_counter.update(value)
        if not aggregate_counter:
            return {"final_answer": None}
        final_answer, _ = aggregate_counter.most_common(1)[0]
        return {"final_answer": final_answer}

    def combine_plan_results(
        self, results: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combines results from the execution plan, extracting the aggregated final answer.

        Args:
            results (Dict[str, Any]): A mapping from task IDs to their outputs.
            inputs (Dict[str, Any]): The original input dictionary.

        Returns:
            Dict[str, Any]: A dictionary with the key 'final_answer' representing the most common answer.
        """
        return results.get("aggregate_most_common", {"final_answer": None})


####################################################
# 3) GetAnswerOperator
####################################################


class GetAnswerOperatorInputs(BaseModel):
    """Input data model for GetAnswerOperator.

    Attributes:
        query (str): The query string.
        responses (List[str]): A list of response strings to process.
    """

    query: str
    responses: List[str]


@register_operator(registry=OperatorRegistryGlobal)
class GetAnswerOperator(Operator[GetAnswerOperatorInputs, Dict[str, Any]]):
    """Operator that processes responses to generate a final answer.

    Typically, only the first LMModule is used for generating the answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="GET_ANSWER",
        description="Processes responses to generate a single answer",
        signature=Signature(input_model=GetAnswerOperatorInputs),
    )

    def forward(self, inputs: GetAnswerOperatorInputs) -> Dict[str, Any]:
        """Generates a final answer by processing the provided responses.

        Args:
            inputs (GetAnswerOperatorInputs): The input containing the query and responses.

        Returns:
            Dict[str, Any]: A dictionary with the key 'final_answer' produced by the LMModule.

        Raises:
            ValueError: If no LMModule is attached.
        """
        prompt_inputs: Dict[str, Any] = inputs.model_dump()
        prompt: str = self.metadata.signature.render_prompt(inputs=prompt_inputs)
        if not self.lm_modules:
            raise ValueError("No LM module is attached to GetAnswerOperator.")
        single_char_answer: str = self.call_lm(
            prompt=prompt, lm=self.lm_modules[0]
        ).strip()
        return {"final_answer": single_char_answer}

    def to_plan(self, inputs: GetAnswerOperatorInputs) -> Optional[ExecutionPlan]:
        """Creates an execution plan to concurrently process each response.

        Args:
            inputs (GetAnswerOperatorInputs): The input containing the query and responses.

        Returns:
            Optional[ExecutionPlan]: An execution plan if LMModules and responses exist; otherwise, None.
        """
        if not self.lm_modules or not inputs.responses:
            return None

        plan: ExecutionPlan = ExecutionPlan()
        for index, response in enumerate(inputs.responses):
            task_id: str = f"get_answer_task_{index}"
            plan.add_task(
                ExecutionTask(
                    task_id=task_id,
                    function=self._process_single_response,
                    inputs={"query": inputs.query, "response": response},
                    dependencies=[],
                )
            )
        return plan

    def _process_single_response(self, *, query: str, response: str) -> str:
        """Processes an individual response to produce an answer.

        Args:
            query (str): The query string.
            response (str): A single response string.

        Returns:
            str: The answer produced by the LMModule, with extraneous whitespace removed.
        """
        prompt_inputs: Dict[str, Any] = {"query": query, "response": response}
        prompt: str = self.metadata.signature.render_prompt(inputs=prompt_inputs)
        answer: str = self.call_lm(prompt=prompt, lm=self.lm_modules[0]).strip()
        return answer

    def combine_plan_results(
        self, results: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combines individual task results by selecting the first non-empty answer.

        Args:
            results (Dict[str, Any]): A mapping from task IDs to their respective responses.
            inputs (Dict[str, Any]): The original input dictionary.

        Returns:
            Dict[str, Any]: A dictionary with the key 'final_answer' set to the first non-empty answer,
            or an empty string if none are found.
        """
        for task_id in sorted(results.keys()):
            answer: str = results[task_id]
            if answer:
                return {"final_answer": answer}
        return {"final_answer": ""}


####################################################
# 4) JudgeSynthesisOperator
####################################################


class JudgeSynthesisInputs(BaseModel):
    """Input data model for JudgeSynthesisOperator.

    Attributes:
        query (str): The query string.
        responses (List[str]): Aggregated list of all ensemble responses.
    """

    query: str
    responses: List[str] = Field(
        ..., description="Aggregated list of all ensemble responses."
    )


class JudgeSynthesisOutputs(BaseModel):
    """Output data model for JudgeSynthesisOperator.

    Attributes:
        final_answer (str): The synthesized final answer.
    """

    final_answer: str


class JudgeSynthesisSignature(Signature):
    """Signature for JudgeSynthesisOperator.

    Attributes:
        prompt_template (str): Template for constructing the LM prompt.
        structured_output (Optional[Type[BaseModel]]): The output model.
        input_model (Type[JudgeSynthesisInputs]): The input data model.
    """

    prompt_template: str = (
        "We have multiple advisors who proposed different answers:\n"
        "{responses}\n"
        "Now, we want to synthesize a single best, final answer to:\n"
        "{query}\n"
        "Explain your reasoning concisely, then provide the single best final answer.\n"
        "Format:\n"
        "Reasoning: <some text>\n"
        "Final Answer: <the single best answer>\n"
    )
    structured_output: Optional[Type[BaseModel]] = JudgeSynthesisOutputs
    input_model: Type[JudgeSynthesisInputs] = JudgeSynthesisInputs


@register_operator(registry=OperatorRegistryGlobal)
class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, Dict[str, Any]]):
    """Operator that synthesizes a final answer with reasoning from multiple responses.

    This operator leverages LMModules to produce a consolidated answer based on aggregated responses.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="JUDGE_SYNTHESIS",
        description="Synthesizes a final answer with reasoning from multiple responses.",
        signature=JudgeSynthesisSignature(),
    )

    def forward(self, inputs: JudgeSynthesisInputs) -> Dict[str, Any]:
        """Synthesizes a final answer from multiple responses using an LMModule.

        Args:
            inputs (JudgeSynthesisInputs): The input containing the query and aggregated responses.

        Returns:
            Dict[str, Any]: A dictionary with keys 'final_answer' and 'reasoning' derived from the LMModule output.

        Raises:
            ValueError: If no LMModules are attached.
        """
        if not self.lm_modules:
            raise ValueError("No LM modules attached to JudgeSynthesisOperator.")
        lm: LMModule = self.lm_modules[0]
        prompt: str = self.metadata.signature.render_prompt(inputs=inputs.model_dump())
        raw_output: str = self.call_lm(prompt=prompt, lm=lm).strip()

        final_answer: str = "Unknown"
        reasoning_lines: List[str] = []
        for line in raw_output.split("\n"):
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            reasoning_lines.append(line)
        reasoning: str = "\n".join(reasoning_lines)
        return {"final_answer": final_answer, "reasoning": reasoning}

    def to_plan(self, inputs: JudgeSynthesisInputs) -> Optional[ExecutionPlan]:
        """Creates an execution plan to synthesize a final answer from multiple responses.

        Args:
            inputs (JudgeSynthesisInputs): The input containing the query and responses.

        Returns:
            Optional[ExecutionPlan]: An execution plan if LMModules and responses exist; otherwise, None.
        """
        if not self.lm_modules or not inputs.responses:
            return None

        plan: ExecutionPlan = ExecutionPlan()
        # Create tasks for processing each individual response.
        for index, response in enumerate(inputs.responses):
            task_id: str = f"judge_single_{index}"
            plan.add_task(
                ExecutionTask(
                    task_id=task_id,
                    function=self._judge_single_response,
                    inputs={"query": inputs.query, "response": response},
                    dependencies=[],
                )
            )
        # Aggregator task to synthesize the final answer.
        plan.add_task(
            ExecutionTask(
                task_id="judge_synthesis_agg",
                function=self._synthesis_responses,
                inputs={},
                dependencies=[
                    f"judge_single_{i}" for i in range(len(inputs.responses))
                ],
            )
        )
        return plan

    def _judge_single_response(self, *, query: str, response: str) -> Dict[str, Any]:
        """Evaluates a single response to produce a partial judgment.

        Args:
            query (str): The query string.
            response (str): A single response string to evaluate.

        Returns:
            Dict[str, Any]: A dictionary containing the original response and its partial judgment.
        """
        prompt_inputs: Dict[str, Any] = {"query": query, "responses": [response]}
        prompt: str = self.metadata.signature.render_prompt(inputs=prompt_inputs)
        partial_output: str = self.call_lm(prompt=prompt, lm=self.lm_modules[0]).strip()
        return {"response": response, "partial_judgment": partial_output}

    def _synthesis_responses(self, **kwargs: Any) -> Dict[str, Any]:
        """Synthesizes the final answer and reasoning from partial judgments.

        Args:
            **kwargs (Any): Task results containing the partial judgments.

        Returns:
            Dict[str, Any]: A dictionary with keys 'final_answer' and 'reasoning' synthesized from the judgments.
        """
        partial_data: List[Any] = list(kwargs.values())
        if not partial_data:
            return {"final_answer": "No responses", "reasoning": ""}
        combined_judgments: str = "\n".join(
            f"{entry['response']}: {entry['partial_judgment']}"
            for entry in partial_data
            if isinstance(entry, dict)
        )
        final_answer: str = "Synthesized answer from partial_judgments"
        return {"final_answer": final_answer, "reasoning": combined_judgments}

    def combine_plan_results(
        self, results: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combines results from the execution plan by extracting the aggregator task's output.

        Args:
            results (Dict[str, Any]): A mapping from task IDs to their outputs.
            inputs (Dict[str, Any]): The original input data.

        Returns:
            Dict[str, Any]: A dictionary with the synthesized 'final_answer' and 'reasoning'.
        """
        return results.get("judge_synthesis_agg", {"final_answer": "Unknown"})


####################################################
# 5) VerifierOperator
####################################################


class VerifierOperatorInputs(BaseModel):
    """Input data model for VerifierOperator.

    Attributes:
        query (str): The query string.
        candidate_answer (str): The candidate answer to verify.
    """

    query: str
    candidate_answer: str


@register_operator(registry=OperatorRegistryGlobal)
class VerifierOperatorOutputs(BaseModel):
    """Output data model for VerifierOperator.

    Attributes:
        verdict (str): The verification verdict.
        explanation (str): Explanation for the verdict.
        revised_answer (Optional[str]): An optional revised answer, if applicable.
    """

    verdict: str
    explanation: str
    revised_answer: Optional[str] = None


class VerifierSignature(Signature):
    """Signature for VerifierOperator.

    Attributes:
        prompt_template (str): The template for the verification prompt.
        structured_output (Optional[Type[BaseModel]]): The output model for verification results.
        input_model (Type[VerifierOperatorInputs]): The input model for verification.
    """

    prompt_template: str = (
        "You are a verifier of correctness.\n"
        "Question: {query}\n"
        "Candidate Answer: {candidate_answer}\n"
        "Please decide if this is correct. Provide:\n"
        "Verdict: <Correct or Incorrect>\n"
        "Explanation: <Your reasoning>\n"
        "Revised Answer (optional): <If you want to provide a corrected version>\n"
    )
    structured_output: Optional[Type[BaseModel]] = VerifierOperatorOutputs
    input_model: Type[VerifierOperatorInputs] = VerifierOperatorInputs


@register_operator(registry=OperatorRegistryGlobal)
class VerifierOperator(Operator[VerifierOperatorInputs, Dict[str, Any]]):
    """Operator that verifies a candidate answer and optionally provides a revised answer.

    Analyzes the candidate answer using an LMModule to produce a verdict, explanation, and possibly a revised answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VERIFIER",
        description="Verifies the correctness of a final answer and may revise it.",
        signature=VerifierSignature(),
    )

    def forward(self, inputs: VerifierOperatorInputs) -> Dict[str, Any]:
        """Verifies the candidate answer and optionally generates a revised answer.

        Args:
            inputs (VerifierOperatorInputs): The input containing the query and candidate answer.

        Returns:
            Dict[str, Any]: A dictionary with keys 'verdict', 'explanation', and 'revised_answer' (if available).

        Raises:
            ValueError: If no LMModules are attached.
        """
        if not self.lm_modules:
            raise ValueError("No LM modules attached to VerifierOperator.")
        lm: LMModule = self.lm_modules[0]
        prompt: str = self.metadata.signature.render_prompt(inputs=inputs.model_dump())
        raw_output: str = self.call_lm(prompt=prompt, lm=lm).strip()

        verdict: str = "Unknown"
        explanation: str = ""
        revised_answer: Optional[str] = None
        for line in raw_output.split("\n"):
            if line.startswith("Verdict:"):
                verdict = line.replace("Verdict:", "").strip()
            elif line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()
            elif line.startswith("Revised Answer:"):
                revised_answer = line.replace("Revised Answer:", "").strip()
        return {
            "verdict": verdict,
            "explanation": explanation,
            "revised_answer": revised_answer,
        }

    def to_plan(self, inputs: VerifierOperatorInputs) -> Optional[ExecutionPlan]:
        """Generates an execution plan for verifying the candidate answer.

        Note:
            This operator does not utilize an execution plan and always returns None.

        Args:
            inputs (VerifierOperatorInputs): The input containing the query and candidate answer.

        Returns:
            Optional[ExecutionPlan]: Always returns None.
        """
        return None
