from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import (
    Operator,
)
from src.ember.core.exceptions import MissingLMModuleError

from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.model.modules.lm import LMModule
from src.ember.core.registry.operator.base._module import ember_field


class VerifierOperatorInputs(BaseModel):
    """Input model for VerifierOperator.

    Attributes:
        query (str): The query string.
        candidate_answer (str): The candidate answer to verify.
    """

    query: str
    candidate_answer: str


class VerifierSignature(Signature):
    """Signature for VerifierOperator defining the verification prompt."""

    prompt_template: str = (
        "You are a verifier of correctness.\n"
        "Question: {query}\n"
        "Candidate Answer: {candidate_answer}\n"
        "Please decide if this is correct. Provide:\n"
        "Verdict: <Correct or Incorrect>\n"
        "Explanation: <Your reasoning>\n"
        "Revised Answer (optional): <If you want to provide a corrected version>\n"
    )


class VerifierOperator(Operator[VerifierOperatorInputs, Dict[str, Any]]):
    """Operator to verify a candidate answer synchronously and optionally suggest revisions."""

    signature: Signature = Signature(
        input_model=VerifierOperatorInputs,
        prompt_template=VerifierSignature().prompt_template,
    )
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        self.lm_module = lm_module

    def forward(self, *, inputs: VerifierOperatorInputs) -> Dict[str, Any]:
        if not self.lm_module:
            raise MissingLMModuleError("No LM module attached to VerifierOperator.")
        rendered_prompt: str = self.signature.render_prompt(inputs=inputs.model_dump())
        raw_output: str = self.lm_module(prompt=rendered_prompt).strip()

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
                verification["revised_answer"] = line.replace(
                    "Revised Answer:", ""
                ).strip()
        return verification
