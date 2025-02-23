from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.core.exceptions import MissingLMModuleError
from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.model.model_modules.lm import LMModule


class VerifierOperatorInputs(BaseModel):
    """Input model for VerifierOperator.

    Attributes:
        query (str): The query string.
        candidate_answer (str): The candidate answer to verify.
    """

    query: str
    candidate_answer: str


class VerifierOperatorOutputs(BaseModel):
    """Typed output model for VerifierOperator.

    Attributes:
        verdict (int): 1 if correct, 0 if incorrect.
        explanation (str): Explanation for the verdict.
        revised_answer (Optional[str]): Optional corrected answer.
    """

    verdict: int = Field(..., description="1 for correct, 0 for incorrect")
    explanation: str
    revised_answer: Optional[str]


class VerifierSignature(Signature):
    """Signature for VerifierOperator defining the verification prompt."""

    prompt_template: str = (
        "You are a verifier of correctness.\n"
        "Question: {query}\n"
        "Candidate Answer: {candidate_answer}\n"
        "Please decide if this is correct. Provide:\n"
        "Verdict: <1 for correct, 0 for incorrect>\n"
        "Explanation: <Your reasoning>\n"
        "Revised Answer (optional): <If you want to provide a corrected version>\n"
    )


class VerifierOperator(Operator[VerifierOperatorInputs, VerifierOperatorOutputs]):
    """Operator to verify a candidate answer and optionally suggest revisions."""

    signature: Signature = Signature(
        input_model=VerifierOperatorInputs,
        prompt_template=VerifierSignature().prompt_template,
    )
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        self.lm_module = lm_module

    def forward(self, *, inputs: VerifierOperatorInputs) -> VerifierOperatorOutputs:
        if not self.lm_module:
            raise MissingLMModuleError("No LM module attached to VerifierOperator.")
        rendered_prompt: str = self.signature.render_prompt(inputs=inputs.model_dump())
        raw_output: str = self.lm_module(prompt=rendered_prompt).strip()

        # Defaults; we parse lines to fill them in
        parsed_output: Dict[str, Any] = {
            "verdict": 0,
            "explanation": "",
            "revised_answer": None,
        }
        for line in raw_output.split("\n"):
            clean_line = line.strip()
            if clean_line.startswith("Verdict:"):
                verdict_value = clean_line.replace("Verdict:", "").strip()
                parsed_output["verdict"] = 1 if verdict_value == "1" else 0
            elif clean_line.startswith("Explanation:"):
                parsed_output["explanation"] = clean_line.replace("Explanation:", "").strip()
            elif clean_line.startswith("Revised Answer:"):
                parsed_output["revised_answer"] = clean_line.replace("Revised Answer:", "").strip()

        return VerifierOperatorOutputs(**parsed_output)
