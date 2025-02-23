from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import Operator

from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.model.model_modules.lm import LMModule
from src.ember.core.registry.operator.base._module import ember_field


class EnsembleOperatorInputs(BaseModel):
    """Input model for EnsembleOperator.

    Attributes:
        query (str): The query string used for prompt rendering.
    """

    query: str


class EnsembleOperator(Operator[EnsembleOperatorInputs, Dict[str, Any]]):
    """Operator that executes parallel calls to multiple LMModules concurrently."""

    signature: Signature = Signature(input_model=EnsembleOperatorInputs)
    lm_modules: List[LMModule]

    def __init__(self, *, lm_modules: List[LMModule]) -> None:
        self.lm_modules = lm_modules

    def forward(self, *, inputs: EnsembleOperatorInputs) -> Dict[str, Any]:
        rendered_prompt: str = self.signature.render_prompt(inputs=inputs.model_dump())
        responses: List[Any] = [lm(prompt=rendered_prompt) for lm in self.lm_modules]
        return {"responses": responses}
