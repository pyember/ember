from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.core.exceptions import MissingLMModuleError

from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.model.modules.lm import LMModule
from src.ember.core.registry.operator.base._module import ember_field


class SelectorJudgeInputs(BaseModel):
    """Input model for SelectorJudgeOperator."""

    query: str
    responses: List[str] = Field(..., description="Aggregated ensemble responses.")


class SelectorJudgeOutputs(BaseModel):
    """Output model for SelectorJudgeOperator."""

    final_answer: str


class SelectorJudgeSignature(Signature):
    """Signature for SelectorJudgeOperator defining the synthesis prompt."""

    prompt_template: str = (
        "We have multiple advisors who proposed different answers:\n"
        "{responses}\n"
        "Now, we want to select the best, final answer to:\n"
        "{query}\n"
        "Explain your reasoning concisely, then provide the single best final answer.\n"
        "Format:\n"
        "Reasoning: <some text>\n"
        "Final Answer: <the single best answer>\n"
    )
    structured_output: Optional[Type[BaseModel]] = SelectorJudgeOutputs
    input_model: Type[BaseModel] = SelectorJudgeInputs


class SelectorJudgeOperator(Operator[SelectorJudgeInputs, Dict[str, Any]]):
    """Operator to select the best, final answer from multiple responses."""

    signature: Signature = SelectorJudgeSignature()
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        self.lm_module = lm_module

    def forward(self, *, inputs: SelectorJudgeInputs) -> Dict[str, Any]:
        rendered_prompt: str = self.signature.render_prompt(inputs=inputs.model_dump())
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
