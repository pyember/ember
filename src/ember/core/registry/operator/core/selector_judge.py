from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from pydantic import Field

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.exceptions import MissingLMModuleError
from ember.core.types import EmberModel

from ember.core.registry.prompt_specification.specification import Specification
from ember.core.registry.model.model_module.lm import LMModule


class SelectorJudgeInputs(EmberModel):
    """Input model for SelectorJudgeOperator."""

    query: str
    responses: List[str] = Field(..., description="Aggregated ensemble responses.")


class SelectorJudgeOutputs(EmberModel):
    """Output model for SelectorJudgeOperator."""

    final_answer: str


class SelectorJudgeOperatorOutputs(EmberModel):
    """Output model for SelectorJudgeOperator.

    Attributes:
        final_answer (str): The selected best final answer.
        reasoning (str): Reasoning behind the selection.
    """

    final_answer: str
    reasoning: str


class SelectorJudgeSpecification(Specification):
    """Specification for SelectorJudgeOperator defining the synthesis prompt."""

    prompt_template: str = (
        "We have multiple advisors who proposed different answers:\n"
        "{responses}\n"
        "Now, we want to select the best, final answer to:\n"
        "{query}\n"
        "Explain your reasoning concisely, then provide the single best final answer.\n"
        "Format:\n"
        "Reasoning: <your reasoning for selecting this answer>\n"
        "Final Answer: <the single best answer>\n"
    )
    structured_output: Optional[Type[EmberModel]] = SelectorJudgeOutputs
    input_model: Type[EmberModel] = SelectorJudgeInputs


class SelectorJudgeOperator(
    Operator[SelectorJudgeInputs, SelectorJudgeOperatorOutputs]
):
    """Operator to select the best, final answer from multiple responses."""

    specification: Specification = SelectorJudgeSpecification()
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        self.lm_module = lm_module

    def forward(self, *, inputs: SelectorJudgeInputs) -> SelectorJudgeOperatorOutputs:
        rendered_prompt: str = self.specification.render_prompt(inputs=inputs)
        if not self.lm_module:
            raise MissingLMModuleError(
                "No LM module attached to SelectorJudgeOperator."
            )
        raw_output: str = self.lm_module(prompt=rendered_prompt).strip()

        # Parse the output for the final answer and reasoning, respectively
        final_answer: str = "Unknown"
        reasoning_lines: List[str] = []
        for line in raw_output.split("\n"):
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            reasoning_lines.append(line)
        reasoning: str = "\n".join(reasoning_lines)
        return {"final_answer": final_answer, "reasoning": reasoning}
