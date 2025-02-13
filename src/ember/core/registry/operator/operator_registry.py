from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ember.core.registry.operator.core.operator_base import Operator, OperatorMetadata
from ember.core.registry.model.modules.lm import LMModule
from ember.core.registry.prompt_signature.signatures import Signature
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine.plan_builder import PlanBuilder


##############################################################################
# Operator Registry and Registration Decorator
##############################################################################
class OperatorRegistry:
    """Global registry for mapping operator codes to operator classes.

    This registry provides methods for registering, retrieving, and instantiating
    operator classes using unique string identifiers.
    """

    def __init__(self) -> None:
        """Initializes an empty OperatorRegistry instance."""
        self._registry: Dict[str, Type[Operator[Any, Any]]] = {}

    def register(
        self, operator_code: str, operator_cls: Type[Operator[Any, Any]]
    ) -> None:
        """Registers an operator class with the specified operator code.

        Args:
            operator_code (str): A unique identifier for the operator.
            operator_cls (Type[Operator[Any, Any]]): The operator class to register.
        """
        self._registry[operator_code] = operator_cls

    def get(self, operator_code: str) -> Optional[Type[Operator[Any, Any]]]:
        """Retrieves the operator class associated with the given operator code.

        Args:
            operator_code (str): The unique identifier for the operator.

        Returns:
            Optional[Type[Operator[Any, Any]]]: The corresponding operator class if found; otherwise, None.
        """
        return self._registry.get(operator_code)

    def create_operator(self, operator_code: str, **params: Any) -> Operator[Any, Any]:
        """Instantiates an operator using its registered code and provided parameters.

        Args:
            operator_code (str): The unique identifier for the operator.
            **params (Any): Additional keyword arguments for the operator's constructor.

        Returns:
            Operator[Any, Any]: An instance of the specified operator.

        Raises:
            ValueError: If no operator is registered with the provided code.
        """
        operator_cls: Optional[Type[Operator[Any, Any]]] = self.get(operator_code)
        if operator_cls is None:
            raise ValueError(f"No operator registered under code '{operator_code}'.")
        return operator_cls(**params)


OPERATOR_REGISTRY_GLOBAL: OperatorRegistry = OperatorRegistry()


def register_operator(
    registry: OperatorRegistry, code: Optional[str] = None
) -> Callable[[Type[Operator[Any, Any]]], Type[Operator[Any, Any]]]:
    """Decorator factory for registering an operator class in a given registry.

    If an explicit code is provided, it is used for registration;
    otherwise, the operator class's metadata.code attribute is utilized.

    Args:
        registry (OperatorRegistry): The registry instance in which to register the operator.
        code (Optional[str]): Optional override code for registration.

    Returns:
        Callable[[Type[Operator[Any, Any]]], Type[Operator[Any, Any]]]:
            A decorator that registers the operator class.
    """

    def decorator(cls: Type[Operator[Any, Any]]) -> Type[Operator[Any, Any]]:
        registration_code: str = code if code is not None else cls.metadata.code
        registry.register(registration_code, cls)
        return cls

    return decorator


##############################################################################
# 1) EnsembleOperator
##############################################################################
class EnsembleOperatorInputs(BaseModel):
    """Input model for EnsembleOperator.

    Attributes:
        query (str): The query string used for prompt rendering.
    """
    query: str


@register_operator(registry=OPERATOR_REGISTRY_GLOBAL)
class EnsembleOperator(Operator[EnsembleOperatorInputs, Dict[str, Any]]):
    """Operator to execute parallel calls to multiple LMModules concurrently.

    This operator renders a prompt from the input and invokes each attached LMModule
    concurrently, aggregating their outputs.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="ENSEMBLE",
        description="Runs an ensemble of models to generate responses",
        signature=Signature(input_model=EnsembleOperatorInputs),
    )

    def forward(self, *, inputs: EnsembleOperatorInputs) -> Dict[str, Any]:
        """Executes synchronous LMModule calls using the rendered prompt.

        Args:
            inputs (EnsembleOperatorInputs): Input data required for prompt rendering.

        Returns:
            Dict[str, Any]: A dictionary with the key 'responses' mapping to the list of LMModule outputs.
        """
        rendered_prompt: str = self.metadata.signature.render_prompt(
            inputs=inputs.model_dump()
        )
        responses: List[Any] = [lm(prompt=rendered_prompt) for lm in self.lm_modules]
        return {"responses": responses}

    def to_plan(self, *, inputs: EnsembleOperatorInputs) -> Optional[XCSGraph]:
        """Builds an execution plan for concurrent LMModule calls.

        The plan creates individual tasks for each LMModule call and aggregates their responses.

        Args:
            inputs (EnsembleOperatorInputs): Input data for constructing the execution plan.

        Returns:
            Optional[XCSGraph]: An execution plan graph if LMModules are present; otherwise, None.
        """
        if not self.lm_modules:
            return None

        global_input: Dict[str, Any] = {
            "prompt": self.metadata.signature.render_prompt(
                inputs=inputs.model_dump()
            )
        }
        plan_builder: PlanBuilder[Dict[str, Any]] = PlanBuilder()

        def create_lm_call_task(lm_instance: LMModule) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
            """Creates a task function that calls the provided LMModule with the global prompt.

            Args:
                lm_instance (LMModule): The LMModule instance to invoke.

            Returns:
                Callable[[Dict[str, Any], Dict[str, Any]], Any]: A task function for the LMModule call.
            """

            def lm_call_task(*, g_in: Dict[str, Any], upstream: Dict[str, Any]) -> Any:
                return lm_instance(prompt=g_in["prompt"])

            return lm_call_task

        for index, lm_instance in enumerate(self.lm_modules):
            plan_builder.add_task(
                name=f"lm_call_{index}",
                function=create_lm_call_task(lm_instance),
            )

        def aggregate_responses(*, g_in: Dict[str, Any], upstream: Dict[str, Any]) -> Dict[str, Any]:
            """Aggregates outputs from LMModule tasks.

            Args:
                g_in (Dict[str, Any]): Global input data (unused in aggregation).
                upstream (Dict[str, Any]): Mapping of task names to their outputs.

            Returns:
                Dict[str, Any]: A dictionary with aggregated responses under the key 'responses'.
            """
            return {"responses": list(upstream.values())}

        plan_builder.add_task(
            name="aggregate_responses",
            function=aggregate_responses,
            depends_on=plan_builder.get_task_names(),
        )

        return plan_builder.build()

    def combine_plan_results(
        self, *, results: Dict[str, Any], inputs: EnsembleOperatorInputs
    ) -> Dict[str, Any]:
        """Combines results from the execution plan into a final aggregated response.

        Args:
            results (Dict[str, Any]): Mapping of task outputs.
            inputs (EnsembleOperatorInputs): The original input instance.

        Returns:
            Dict[str, Any]: A dictionary with the key 'responses' containing the aggregated outputs.
        """
        aggregated: Optional[Any] = results.get("aggregate_responses")
        if aggregated is not None:
            return aggregated
        return {"responses": list(results.values())}


##############################################################################
# 2) MostCommonOperator
##############################################################################
class MostCommonOperatorInputs(BaseModel):
    """Input model for MostCommonOperator.

    Attributes:
        query (str): The query string.
        responses (List[str]): A list of response strings.
    """
    query: str
    responses: List[str]


@register_operator(registry=OPERATOR_REGISTRY_GLOBAL)
class MostCommonOperator(Operator[MostCommonOperatorInputs, Dict[str, Any]]):
    """Operator to determine the most common answer among provided responses.

    This operator offers both synchronous evaluation and an execution plan to concurrently
    count and aggregate responses.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="MOST_COMMON",
        description="Determines the most common answer from responses",
        signature=Signature(input_model=MostCommonOperatorInputs),
    )

    def forward(self, *, inputs: MostCommonOperatorInputs) -> Dict[str, Any]:
        """Determines the most frequent response synchronously.

        Args:
            inputs (MostCommonOperatorInputs): Input data containing the query and responses.

        Returns:
            Dict[str, Any]: A dictionary with 'final_answer' mapped to the most common response.
        """
        if not inputs.responses:
            return {"final_answer": None}
        response_counts: Counter = Counter(inputs.responses)
        most_common_answer, _ = response_counts.most_common(1)[0]
        return {"final_answer": most_common_answer}

    def to_plan(self, *, inputs: MostCommonOperatorInputs) -> Optional[XCSGraph]:
        """Constructs an execution plan to count response occurrences concurrently.

        Args:
            inputs (MostCommonOperatorInputs): Input data containing the query and responses.

        Returns:
            Optional[XCSGraph]: An execution plan graph if responses are present; otherwise, None.
        """
        if not inputs.responses:
            return None

        global_input: Dict[str, Any] = {"responses": inputs.responses}
        plan_builder: PlanBuilder[Dict[str, Any]] = PlanBuilder()

        def create_count_task(response: str) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, int]]:
            """Creates a task function that counts an individual response.

            Args:
                response (str): The response string to count.

            Returns:
                Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, int]]:
                    A task function returning a count dictionary for the response.
            """

            def count_task(*, g_in: Dict[str, Any], upstream: Dict[str, Any]) -> Dict[str, int]:
                return {response: 1}

            return count_task

        for idx, response in enumerate(global_input["responses"]):
            plan_builder.add_task(name=f"count_{idx}", function=create_count_task(response))

        plan_builder.add_task(
            name="aggregate_counts",
            function=self._agg_counts_list,
            depends_on=plan_builder.get_task_names(),
        )

        return plan_builder.build()

    def _agg_counts_list(self, *, upstream: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregates counts from multiple tasks to determine the most common response.

        Args:
            upstream (Dict[str, Any]): Mapping of task names to count outputs.

        Returns:
            Dict[str, Any]: A dictionary with 'final_answer' mapped to the most common response,
            or None if no counts were aggregated.
        """
        aggregated_counts: Counter = Counter()
        for count_dict in upstream.values():
            aggregated_counts.update(count_dict)
        if not aggregated_counts:
            return {"final_answer": None}
        most_common_answer, _ = aggregated_counts.most_common(1)[0]
        return {"final_answer": most_common_answer}

    def combine_plan_results(
        self, *, results: Dict[str, Any], inputs: MostCommonOperatorInputs
    ) -> Dict[str, Any]:
        """Combines outputs from concurrent tasks to yield the most common answer.

        Args:
            results (Dict[str, Any]): A mapping of task outputs.
            inputs (MostCommonOperatorInputs): The original input instance.

        Returns:
            Dict[str, Any]: A dictionary with 'final_answer' derived from aggregated counts.
        """
        aggregated: Optional[Any] = results.get("aggregate_counts")
        if aggregated is not None:
            return aggregated

        combined_counts: Counter = Counter()
        for task_result in results.values():
            if isinstance(task_result, dict):
                combined_counts.update(task_result)
        if not combined_counts:
            return {"final_answer": None}
        common_answer, _ = combined_counts.most_common(1)[0]
        return {"final_answer": common_answer}


##############################################################################
# 3) GetAnswerOperator
##############################################################################
class GetAnswerOperatorInputs(BaseModel):
    """Input model for GetAnswerOperator.

    Attributes:
        query (str): The query string.
        responses (List[str]): A list of response strings.
    """
    query: str
    responses: List[str]


@register_operator(registry=OPERATOR_REGISTRY_GLOBAL)
class GetAnswerOperator(Operator[GetAnswerOperatorInputs, Dict[str, Any]]):
    """Operator to process responses and generate a final answer using an LMModule.

    Typically, only the first LMModule is utilized to synthesize the final answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="GET_ANSWER",
        description="Processes responses to generate a single answer",
        signature=Signature(input_model=GetAnswerOperatorInputs),
    )

    def forward(self, *, inputs: GetAnswerOperatorInputs) -> Dict[str, Any]:
        """Generates a final answer using the first available LMModule.

        Args:
            inputs (GetAnswerOperatorInputs): Input data containing the query and responses.

        Returns:
            Dict[str, Any]: A dictionary with 'final_answer' containing the processed answer.

        Raises:
            ValueError: If no LMModule is attached to the operator.
        """
        rendered_prompt: str = self.metadata.signature.render_prompt(
            inputs=inputs.model_dump()
        )
        if not self.lm_modules:
            raise ValueError("No LM module attached to GetAnswerOperator.")
        final_answer: str = self.call_lm(prompt=rendered_prompt, lm=self.lm_modules[0]).strip()
        return {"final_answer": final_answer}

    def to_plan(self, *, inputs: GetAnswerOperatorInputs) -> Optional[XCSGraph]:
        """Constructs an execution plan for processing responses concurrently.

        Args:
            inputs (GetAnswerOperatorInputs): Input data containing the query and responses.

        Returns:
            Optional[XCSGraph]: An execution plan graph if conditions are met; otherwise, None.
        """
        if not self.lm_modules or not inputs.responses:
            return None

        global_input: Dict[str, Any] = {
            "prompt": self.metadata.signature.render_prompt(inputs=inputs.model_dump())
        }
        plan_builder: PlanBuilder[Dict[str, Any]] = PlanBuilder()

        def create_process_task(response: str) -> Callable[[Dict[str, Any], Dict[str, Any]], str]:
            """Creates a task function that processes an individual response.

            Args:
                response (str): A response string to process.

            Returns:
                Callable[[Dict[str, Any], Dict[str, Any]], str]:
                    A task function that processes the response and returns the processed output.
            """

            def process_task(*, g_in: Dict[str, Any], upstream: Dict[str, Any]) -> str:
                return self._process_single_response(prompt=g_in["prompt"], response=response)

            return process_task

        for i, response in enumerate(inputs.responses):
            plan_builder.add_task(name=f"process_{i}", function=create_process_task(response))

        def select_first_nonempty(
            *, g_in: Dict[str, Any], upstream: Dict[str, Any]
        ) -> Dict[str, str]:
            """Selects and returns the first non-empty processed response.

            Args:
                g_in (Dict[str, Any]): Global input data (unused).
                upstream (Dict[str, Any]): Mapping of task outputs.

            Returns:
                Dict[str, str]: A dictionary with 'final_answer' mapped to the first non-empty output,
                or an empty string if none are found.
            """
            for output in upstream.values():
                if output:
                    return {"final_answer": output}
            return {"final_answer": ""}

        plan_builder.add_task(
            name="select_first",
            function=select_first_nonempty,
            depends_on=plan_builder.get_task_names(),
        )

        return plan_builder.build()

    def _process_single_response(self, *, prompt: str, response: str) -> str:
        """Processes a single response by stripping excess whitespace.

        Args:
            prompt (str): The prompt that was used.
            response (str): The response string to process.

        Returns:
            str: The processed response.
        """
        return response.strip()

    def combine_plan_results(
        self, *, results: Dict[str, Any], inputs: GetAnswerOperatorInputs
    ) -> Dict[str, Any]:
        """Combines outputs from the execution plan to yield the final answer.

        Args:
            results (Dict[str, Any]): Mapping of task outputs.
            inputs (GetAnswerOperatorInputs): The original input instance.

        Returns:
            Dict[str, Any]: A dictionary with 'final_answer' selected from the first non-empty response.
        """
        selected: Optional[Any] = results.get("select_first")
        if selected is not None:
            return selected

        for result in results.values():
            if isinstance(result, dict) and result.get("final_answer"):
                return result
        return {"final_answer": ""}


##############################################################################
# 4) JudgeSynthesisOperator
##############################################################################
class JudgeSynthesisInputs(BaseModel):
    """Input model for JudgeSynthesisOperator.

    Attributes:
        query (str): The query string.
        responses (List[str]): Aggregated ensemble responses.
    """
    query: str
    responses: List[str] = Field(..., description="Aggregated ensemble responses.")


class JudgeSynthesisOutputs(BaseModel):
    """Output model for JudgeSynthesisOperator.

    Attributes:
        final_answer (str): The synthesized final answer.
    """
    final_answer: str


class JudgeSynthesisSignature(Signature):
    """Signature for JudgeSynthesisOperator, defining the synthesis prompt template.

    Attributes:
        prompt_template (str): Template for rendering the synthesis prompt.
        structured_output (Optional[Type[BaseModel]]): Optional output model.
        input_model (Type[BaseModel]): The input model type.
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


@register_operator(registry=OPERATOR_REGISTRY_GLOBAL)
class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, Dict[str, Any]]):
    """Operator to synthesize a final answer with reasoning from multiple responses.

    This operator processes multiple responses via LMModule calls and aggregates them to determine
    the final answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="JUDGE_SYNTHESIS",
        description="Synthesizes a final answer with reasoning from multiple responses",
        signature=JudgeSynthesisSignature(),
    )

    def forward(self, *, inputs: JudgeSynthesisInputs) -> Dict[str, Any]:
        """Synthesizes a final answer by combining multiple ensemble responses.

        Args:
            inputs (JudgeSynthesisInputs): Input containing the query and ensemble responses.

        Returns:
            Dict[str, Any]: A dictionary with keys 'final_answer' and 'reasoning'.

        Raises:
            ValueError: If no LMModule is attached to the operator.
        """
        if not self.lm_modules:
            raise ValueError("No LM modules attached to JudgeSynthesisOperator.")
        lm_module: LMModule = self.lm_modules[0]
        rendered_prompt: str = self.metadata.signature.render_prompt(
            inputs=inputs.model_dump()
        )
        raw_output: str = self.call_lm(prompt=rendered_prompt, lm=lm_module).strip()
        final_answer: str = "Unknown"
        reasoning_lines: List[str] = []
        for line in raw_output.split("\n"):
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            reasoning_lines.append(line)
        reasoning: str = "\n".join(reasoning_lines)
        return {"final_answer": final_answer, "reasoning": reasoning}

    def to_plan(self, *, inputs: JudgeSynthesisInputs) -> Optional[XCSGraph]:
        """Builds an execution plan to judge responses and synthesize a final answer.

        The plan creates individual judgement tasks for each response and aggregates them in a synthesis task.

        Args:
            inputs (JudgeSynthesisInputs): Input containing the query and ensemble responses.

        Returns:
            Optional[XCSGraph]: An execution plan graph if LMModules and responses are available; otherwise, None.
        """
        if not self.lm_modules or not inputs.responses:
            return None

        global_input: Dict[str, Any] = {"query": inputs.query, "responses": inputs.responses}
        plan_builder: PlanBuilder[Dict[str, Any]] = PlanBuilder()

        def create_judge_task(response: str) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
            """Creates a task function to judge an individual ensemble response.

            Args:
                response (str): A single ensemble response.

            Returns:
                Callable[[Dict[str, Any], Dict[str, Any]], Any]:
                    A task function that returns the judgement for the provided response.
            """

            def judge_task(*, g_in: Dict[str, Any], upstream: Dict[str, Any]) -> Any:
                return self._judge_single_response(query=g_in["query"], response=response)

            return judge_task

        for index, response in enumerate(global_input["responses"]):
            plan_builder.add_task(name=f"judge_{index}", function=create_judge_task(response))

        def synthesize_task(*, g_in: Dict[str, Any], upstream: Dict[str, Any]) -> Any:
            """Aggregates judgement outputs to synthesize the final answer."""
            return self._synthesis_responses(upstream=upstream)

        plan_builder.add_task(
            name="synthesize",
            function=synthesize_task,
            depends_on=plan_builder.get_task_names(),
        )

        return plan_builder.build()

    def combine_plan_results(
        self, *, results: Dict[str, Any], inputs: JudgeSynthesisInputs
    ) -> Dict[str, Any]:
        """Aggregates outputs from judgement tasks to yield the final synthesized answer.

        Args:
            results (Dict[str, Any]): Mapping of task outputs.
            inputs (JudgeSynthesisInputs): The original input instance.

        Returns:
            Dict[str, Any]: A dictionary with 'final_answer' derived from the aggregated synthesis task output.
        """
        aggregated: Optional[Any] = results.get("synthesize")
        if aggregated is not None:
            return aggregated
        return {"final_answer": "Unknown"}


##############################################################################
# 5) VerifierOperator
##############################################################################
class VerifierOperatorInputs(BaseModel):
    """Input model for VerifierOperator.

    Attributes:
        query (str): The query string.
        candidate_answer (str): The candidate answer to verify.
    """
    query: str
    candidate_answer: str


class VerifierSignature(Signature):
    """Signature for VerifierOperator that defines a verification prompt.

    Attributes:
        prompt_template (str): Template for constructing the verification prompt.
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


@register_operator(registry=OPERATOR_REGISTRY_GLOBAL)
class VerifierOperator(Operator[VerifierOperatorInputs, Dict[str, Any]]):
    """Operator to verify a candidate answer synchronously and optionally suggest a revision.

    This operator operates synchronously without constructing an execution plan.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VERIFIER",
        description="Verifies the correctness of a final answer and may revise it",
        signature=VerifierSignature(),
    )

    def forward(self, *, inputs: VerifierOperatorInputs) -> Dict[str, Any]:
        """Verifies a candidate answer synchronously.

        Args:
            inputs (VerifierOperatorInputs): Input data containing the query and candidate answer.

        Returns:
            Dict[str, Any]: A dictionary with keys 'verdict', 'explanation', and 'revised_answer'.

        Raises:
            ValueError: If no LMModule is attached to the operator.
        """
        if not self.lm_modules:
            raise ValueError("No LM modules attached to VerifierOperator.")
        lm_module: LMModule = self.lm_modules[0]
        rendered_prompt: str = self.metadata.signature.render_prompt(
            inputs=inputs.model_dump()
        )
        raw_output: str = self.call_lm(prompt=rendered_prompt, lm=lm_module).strip()
        verification: Dict[str, Any] = {
            "verdict": "Unknown",
            "explanation": "",
            "revised_answer": None,
        }
        for line in raw_output.split("\n"):
            if line.startswith("Verdict:"):
                verification["verdict"] = line.replace("Verdict:", "").strip()
            elif line.startswith("Explanation:"):
                verification["explanation"] = line.replace("Explanation:", "").strip()
            elif line.startswith("Revised Answer:"):
                verification["revised_answer"] = line.replace("Revised Answer:", "").strip()
        return verification

    def to_plan(self, *, inputs: VerifierOperatorInputs) -> Optional[XCSGraph]:
        """Since VerifierOperator runs synchronously, no execution plan is constructed.

        Args:
            inputs (VerifierOperatorInputs): Input data containing the query and candidate answer.

        Returns:
            None.
        """
        return None
