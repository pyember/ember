from __future__ import annotations

from typing import List, Optional, Type
from pydantic import BaseModel, Field

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.core.exceptions import MissingLMModuleError

from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.model.model_modules.lm import LMModule
from src.ember.core.registry.operator.base._module import ember_field


class JudgeSynthesisInputs(BaseModel):
    """Input model for JudgeSynthesisOperator.

    Attributes:
        query (str): The query text.
        responses (List[str]): Aggregated ensemble responses.
    """

    query: str
    responses: List[str] = Field(..., description="Aggregated ensemble responses.")


class JudgeSynthesisOutputs(BaseModel):
    """Output model for JudgeSynthesisOperator.

    Attributes:
        final_answer (str): Synthetically combined best final answer.
        reasoning (str): Rationale behind the combined answer choice.
    """

    final_answer: str
    reasoning: str


class JudgeSynthesisSignature(Signature):
    """Signature for JudgeSynthesisOperator defining the synthesis prompt."""

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
    input_model: Type[BaseModel] = JudgeSynthesisInputs


class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Operator to synthesize a final answer and reasoning from multiple responses."""

    signature: Signature = JudgeSynthesisSignature()
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        """Initialize the synthesis judge with a language model module."""
        self.lm_module = lm_module

    def forward(self, *, inputs: JudgeSynthesisInputs) -> JudgeSynthesisOutputs:
        if not self.lm_module:
            raise MissingLMModuleError("No LM module attached to JudgeSynthesisOperator.")

        rendered_prompt: str = self.signature.render_prompt(inputs=inputs.model_dump())
        raw_output: str = self.lm_module(prompt=rendered_prompt).strip()

        final_answer = "Unknown"
        reasoning_lines: List[str] = []

        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            reasoning_lines.append(line)

        reasoning = "\n".join(reasoning_lines)

        return JudgeSynthesisOutputs(
            final_answer=final_answer,
            reasoning=reasoning,
        )
