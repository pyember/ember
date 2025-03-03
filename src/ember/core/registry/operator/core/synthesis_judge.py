from __future__ import annotations

from typing import List, Optional, Type
from pydantic import Field

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.exceptions import MissingLMModuleError
from ember.core.types.ember_model import EmberModel

from ember.core.registry.prompt_specification.specification import Specification
from ember.core.registry.model.model_module.lm import LMModule


class JudgeSynthesisInputs(EmberModel):
    """Input model for JudgeSynthesisOperator.

    Attributes:
        query (str): The query text.
        responses (List[str]): Aggregated ensemble responses.
    """

    query: str
    responses: List[str] = Field(..., description="Aggregated ensemble responses.")


class JudgeSynthesisOutputs(EmberModel):
    """Output model for JudgeSynthesisOperator.

    Attributes:
        final_answer (str): Synthetically combined best final answer.
        reasoning (str): Rationale behind the combined answer choice.
    """

    final_answer: str
    reasoning: str


class JudgeSynthesisSpecification(Specification):
    """Specification for JudgeSynthesisOperator defining the synthesis prompt."""

    prompt_template: str = (
        "We have multiple advisors who proposed different answers:\n"
        "{responses}\n"
        "Now, we want to synthesize a single best, final answer to:\n"
        "{query}\n"
        "Explain your reasoning concisely, then provide the single best final answer.\n"
        "Format:\n"
        "Reasoning: <your reasoning for synthesizing this answer in this way>\n"
        "Final Answer: <the single best answer>\n"
    )
    structured_output: Optional[Type[EmberModel]] = JudgeSynthesisOutputs
    input_model: Type[EmberModel] = JudgeSynthesisInputs


class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Operator to synthesize a final answer and reasoning from multiple responses."""

    specification: Specification = JudgeSynthesisSpecification()
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        """Initialize the synthesis judge with a language model module."""
        self.lm_module = lm_module

    def forward(self, *, inputs: JudgeSynthesisInputs) -> JudgeSynthesisOutputs:
        if not self.lm_module:
            raise MissingLMModuleError(
                "No LM module attached to JudgeSynthesisOperator."
            )

        rendered_prompt: str = self.specification.render_prompt(inputs=inputs)
        raw_output: str = self.lm_module(prompt=rendered_prompt).strip()

        # Parse the response to extract reasoning and final answer
        final_answer = "Unknown"
        reasoning_lines: List[str] = []
        in_reasoning_section = False

        for line in raw_output.splitlines():
            line = line.strip()

            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            elif line.startswith("Reasoning:"):
                in_reasoning_section = True
                reasoning_part = line.replace("Reasoning:", "").strip()
                if reasoning_part:
                    reasoning_lines.append(reasoning_part)
            elif in_reasoning_section:
                reasoning_lines.append(line)

        reasoning = "\n".join(reasoning_lines)

        return {"final_answer": final_answer, "reasoning": reasoning}
