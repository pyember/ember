from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import (
    Operator,
)
from src.ember.core.exceptions import MissingLMModuleError
from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.model.modules.lm import LMModule
from src.ember.core.registry.operator.base._module import ember_field


class GetAnswerOperatorInputs(BaseModel):
    """Input model for GetAnswerOperator.

    Attributes:
        query (str): The query string.
        response (str): The response string.
    """

    query: str
    response: str


class GetAnswerOperator(Operator[GetAnswerOperatorInputs, Dict[str, Any]]):
    """Operator to process responses and generate a final answer using an LMModule."""

    signature: Signature = Signature(
        input_model=GetAnswerOperatorInputs,
        prompt_template=(
            "GetAnswer Prompt:\n"
            "Query: {query}\n"
            "Previous Response: {response}"
        ),
    )
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        self.lm_module = lm_module

    def forward(self, *, inputs: GetAnswerOperatorInputs) -> Dict[str, Any]:
        rendered_prompt: str = self.signature.render_prompt(inputs=inputs.model_dump())
        if not self.lm_module:
            raise MissingLMModuleError("No LM module attached to GetAnswerOperator.")
        final_answer: str = self.lm_module(prompt=rendered_prompt)
        return {"final_answer": final_answer}
