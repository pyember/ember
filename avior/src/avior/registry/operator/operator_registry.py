from typing import Any, Dict, Optional, List, TypeVar, Type
from collections import Counter
from pydantic import BaseModel, Field

from src.avior.registry.operator.operator_base import (
    Operator,
    OperatorMetadata,
    OperatorType,
    LMModule,
)
from src.avior.registry.prompt_signature.signatures import Signature
from src.avior.core.scheduler import ExecutionPlan, ExecutionTask

class OperatorRegistry:
    """
    A global registry that maps string 'op_code' to a particular Operator class.
    Example usage:
      OperatorRegistry.register("ENSEMBLE", EnsembleOperator)
      ...
    """
    def __init__(self):
        self._registry = {}

    def register(self, code: str, operator_cls: Any):
        self._registry[code] = operator_cls

    def get(self, code: str) -> Optional[Any]:
        return self._registry.get(code)

# For typed usage
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out")

#####################################################
# 1) EnsembleOperator
#####################################################

class EnsembleOperatorInputs(BaseModel):
    query: str

class EnsembleOperator(Operator[EnsembleOperatorInputs, Dict[str, Any]]):
    """
    Parallel calls to multiple LMModules. Each LMModule processes
    the same prompt, producing an array of responses.
    """
    metadata = OperatorMetadata(
        code="ENSEMBLE",
        description="Runs an ensemble of models to generate responses",
        operator_type=OperatorType.FAN_OUT,
        signature=Signature(
            required_inputs=["query"],
            input_model=EnsembleOperatorInputs,
        ),
    )

    @classmethod
    def build_inputs(cls, **fields: Any) -> BaseModel:
        return cls.metadata.signature.input_model(**fields)

    def forward(self, inputs: EnsembleOperatorInputs) -> Dict[str, Any]:
        prompt = self.build_prompt(inputs.model_dump())
        responses = [lm(prompt) for lm in self.lm_modules]
        return {"responses": responses}

    def to_plan(self, inputs: EnsembleOperatorInputs) -> Optional[ExecutionPlan]:
        """
        Creates a task per LMModule for concurrency. We skip dependencies because
        all tasks can run in parallel.
        """
        if not self.lm_modules:
            return None
        prompt = self.build_prompt(inputs.model_dump())
        plan = ExecutionPlan()
        for i, lm in enumerate(self.lm_modules):
            task_id = f"ensemble_task_{i}"
            plan.add_task(ExecutionTask(
                task_id=task_id,
                function=self._lm_call_wrapper,
                inputs={"prompt": prompt, "lm": lm},
                dependencies=[],
            ))
        return plan

    def _lm_call_wrapper(self, prompt: str, lm: LMModule) -> str:
        return lm(prompt)

    def combine_plan_results(
        self, results: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Any:
        # Sort tasks by ID for consistency
        sorted_responses = [results[tid] for tid in sorted(results.keys())]
        return {"responses": sorted_responses}

#####################################################
# 2) MostCommonOperator
#####################################################

class MostCommonOperatorInputs(BaseModel):
    query: str
    responses: List[str]

class MostCommonOperator(Operator[MostCommonOperatorInputs, Dict[str, Any]]):
    metadata = OperatorMetadata(
        code="MOST_COMMON",
        description="Determines the most common answer from responses",
        operator_type=OperatorType.FAN_IN,
        signature=Signature(
            required_inputs=["query", "responses"],
            input_model=MostCommonOperatorInputs,
        ),
    )

    @classmethod
    def build_inputs(cls, **fields: Any) -> BaseModel:
        return cls.metadata.signature.input_model(**fields)

    def forward(self, inputs: MostCommonOperatorInputs) -> Dict[str, Any]:
        if not inputs.responses:
            return {"final_answer": None}
        counts = Counter(inputs.responses)
        [(final_answer, _)] = counts.most_common(1)
        return {"final_answer": final_answer}

    def to_plan(self, inputs: MostCommonOperatorInputs) -> Optional[ExecutionPlan]:
        """
        For demonstration, we'll add concurrency:
          - One task per response that returns {response:1},
          - A final aggregator task that sums and picks the most common.
        """
        if not inputs.responses:
            return None

        plan = ExecutionPlan()
        # 1) Create a counting task per response
        for i, r in enumerate(inputs.responses):
            task_id = f"count_task_{i}"
            plan.add_task(ExecutionTask(
                task_id=task_id,
                function=self._count_single_response,
                inputs={"response": r},
                dependencies=[]
            ))
        # 2) Create aggregator task
        agg_id = "aggregate_most_common"
        plan.add_task(ExecutionTask(
            task_id=agg_id,
            function=self._aggregate_counts,
            inputs={},
            dependencies=[f"count_task_{i}" for i in range(len(inputs.responses))]
        ))
        return plan

    def _count_single_response(self, response: str) -> Dict[str, int]:
        """A trivial function returning a map of {response: 1}."""
        return {response: 1}

    def _aggregate_counts(self, **kwargs) -> Dict[str, Any]:
        big_count = Counter()
        for v in kwargs.values():
            if isinstance(v, dict):
                big_count.update(v)
        if not big_count:
            return {"final_answer": None}
        [(final_answer, _)] = big_count.most_common(1)
        return {"final_answer": final_answer}

    def combine_plan_results(self, results: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        # We look for the final aggregator task's result
        return results.get("aggregate_most_common", {"final_answer": None})

#####################################################
# 3) GetAnswerOperator
#####################################################

class GetAnswerOperatorInputs(BaseModel):
    query: str
    responses: List[str]

class GetAnswerOperator(Operator[GetAnswerOperatorInputs, Dict[str, Any]]):
    metadata = OperatorMetadata(
        code="GET_ANSWER",
        description="Processes the response to generate a single character answer",
        operator_type=OperatorType.RECURRENT,
        signature=Signature(
            required_inputs=["query", "responses"],
            input_model=GetAnswerOperatorInputs,
        ),
    )

    @classmethod
    def build_inputs(cls, **fields: Any) -> BaseModel:
        return cls.metadata.signature.input_model(**fields)

    def forward(self, inputs: GetAnswerOperatorInputs) -> Dict[str, Any]:
        # By default, we only take the first response
        response = inputs.responses[0] if inputs.responses else ""
        prompt_inputs = inputs.model_dump()
        prompt_inputs["response"] = response
        prompt = self.build_prompt(prompt_inputs)
        if not self.lm_modules:
            raise ValueError("No LM module is attached to GetAnswerOperator.")
        single_char_answer = self.call_lm(prompt, self.lm_modules[0]).strip()
        return {"final_answer": single_char_answer}

    def to_plan(self, inputs: GetAnswerOperatorInputs) -> Optional[ExecutionPlan]:
        """
        Example concurrency: each response => separate task. We then pick the first non-empty answer.
        """
        if not self.lm_modules:
            return None
        if not inputs.responses:
            return None

        plan = ExecutionPlan()
        for i, resp in enumerate(inputs.responses):
            task_id = f"get_answer_task_{i}"
            plan.add_task(ExecutionTask(
                task_id=task_id,
                function=self._process_single_response,
                inputs={"query": inputs.query, "response": resp},
                dependencies=[]
            ))
        return plan

    def _process_single_response(self, query: str, response: str) -> str:
        prompt_inputs = {"query": query, "response": response}
        prompt = self.build_prompt(prompt_inputs)
        answer = self.call_lm(prompt, self.lm_modules[0]).strip()
        return answer

    def combine_plan_results(self, results: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        for task_id in sorted(results.keys()):
            ans = results[task_id]
            if ans:
                return {"final_answer": ans}
        return {"final_answer": ""}

#####################################################
# 4) JudgeSynthesisOperator
#####################################################

class JudgeSynthesisInputs(BaseModel):
    query: str
    responses: List[str] = Field(..., description="Aggregated list of all ensemble responses.")

class JudgeSynthesisOutputs(BaseModel):
    final_answer: str

class JudgeSynthesisSignature(Signature):
    required_inputs: List[str] = ["query", "responses"]
    prompt_template: str = (
        "We have multiple advisors who proposed different answers:\n"
        "{responses}\n"
        "Now, we want to Synthesize a single best, final answer to:\n"
        "{query}\n"
        "Explain your reasoning concisely, then provide the single best final answer.\n"
        "Format:\n"
        "Reasoning: <some text>\n"
        "Final Answer: <the single best answer>\n"
    )
    structured_output: Optional[Type[BaseModel]] = JudgeSynthesisOutputs
    input_model: Type[JudgeSynthesisInputs] = JudgeSynthesisInputs

class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, Dict[str, Any]]):
    metadata = OperatorMetadata(
        code="JUDGE_SYNTHESIS",
        description="Takes multiple responses and produces one final answer with reasoning.",
        operator_type=OperatorType.FAN_IN,
        signature=JudgeSynthesisSignature(),
    )

    def forward(self, inputs: JudgeSynthesisInputs) -> Dict[str, Any]:
        if not self.lm_modules:
            raise ValueError("No LM modules attached to JudgeSynthesisOperator.")
        lm = self.lm_modules[0]
        prompt = self.build_prompt(inputs.model_dump())
        raw_output = self.call_lm(prompt, lm).strip()

        final_answer = "Unknown"
        reasoning_lines = []
        for line in raw_output.split("\n"):
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            reasoning_lines.append(line)

        return {
            "final_answer": final_answer,
            "reasoning": "\n".join(reasoning_lines),
        }

    def to_plan(self, inputs: JudgeSynthesisInputs) -> Optional[ExecutionPlan]:
        """
        Demonstrates concurrency: judge each response individually, then do
        a final aggregator step to unify them into a single final answer.
        """
        if not self.lm_modules:
            return None
        if not inputs.responses:
            return None

        plan = ExecutionPlan()
        # 1) tasks for each response
        for i, resp in enumerate(inputs.responses):
            task_id = f"judge_single_{i}"
            plan.add_task(ExecutionTask(
                task_id=task_id,
                function=self._judge_single_response,
                inputs={"query": inputs.query, "response": resp},
                dependencies=[]
            ))
        # 2) final aggregator
        plan.add_task(ExecutionTask(
            task_id="judge_synthesis_agg",
            function=self._synthesis_responses,
            inputs={},
            dependencies=[f"judge_single_{i}" for i in range(len(inputs.responses))]
        ))
        return plan

    def _judge_single_response(self, query: str, response: str) -> Dict[str, Any]:
        prompt_inputs = {"query": query, "responses": [response]}
        prompt = self.build_prompt(prompt_inputs)
        partial_out = self.call_lm(prompt, self.lm_modules[0]).strip()
        return {"response": response, "partial_judgment": partial_out}

    def _synthesis_responses(self, **kwargs) -> Dict[str, Any]:
        partial_data = list(kwargs.values())
        if not partial_data:
            return {"final_answer": "No responses", "reasoning": ""}

        combined_judgments = "\n".join(f"{d['response']}: {d['partial_judgment']}" for d in partial_data if isinstance(d, dict))
        final_answer = "Synthesized answer from partial_judgments"
        reasoning = combined_judgments
        return {"final_answer": final_answer, "reasoning": reasoning}

    def combine_plan_results(self, results: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        return results.get("judge_synthesis_agg", {"final_answer": "Unknown"})

#####################################################
# 5) VerifierOperator
#####################################################

class VerifierOperatorInputs(BaseModel):
    query: str
    candidate_answer: str

class VerifierOperatorOutputs(BaseModel):
    verdict: str
    explanation: str
    revised_answer: Optional[str] = None

class VerifierSignature(Signature):
    required_inputs: List[str] = ["query", "candidate_answer"]
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

class VerifierOperator(Operator[VerifierOperatorInputs, Dict[str, Any]]):
    metadata = OperatorMetadata(
        code="VERIFIER",
        description="Verifies correctness of a final answer and possibly revises it.",
        operator_type=OperatorType.RECURRENT,
        signature=VerifierSignature(),
    )

    def forward(self, inputs: VerifierOperatorInputs) -> Dict[str, Any]:
        if not self.lm_modules:
            raise ValueError("No LM modules attached to VerifierOperator.")
        lm = self.lm_modules[0]
        prompt = self.build_prompt(inputs.model_dump())
        raw_output = self.call_lm(prompt, lm).strip()

        verdict = "Unknown"
        explanation = ""
        revised_answer = None
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
        """
        Typically, we have only one candidate_answer, so concurrency is not needed.
        """
        return None

#####################################################
# Register everything
#####################################################

_registry_instance = OperatorRegistry()
_registry_instance.register("ENSEMBLE", EnsembleOperator)
_registry_instance.register("MOST_COMMON", MostCommonOperator)
_registry_instance.register("GET_ANSWER", GetAnswerOperator)
_registry_instance.register("JUDGE_SYNTHESIS", JudgeSynthesisOperator)
_registry_instance.register("VERIFIER", VerifierOperator)

OperatorRegistryGlobal = _registry_instance
