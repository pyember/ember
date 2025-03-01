from __future__ import annotations

from typing import Any, Dict, List, TypeVar
from ember.core.types import EmberModel

from ember.core.registry.operator.base.operator_base import Operator

from ember.core.registry.prompt_signature.signatures import Signature
from ember.core.registry.model.model_module.lm import LMModule


class EnsembleOperatorInputs(EmberModel):
    """Input model for EnsembleOperator.

    Attributes:
        query (str): The query string used for prompt rendering.
    """

    query: str


class EnsembleOperatorOutputs(EmberModel):
    """Output model for EnsembleOperator.

    Attributes:
        responses (List[str]): List of LM responses to the prompt.
    """

    responses: List[str]


class EnsembleOperator(Operator[EnsembleOperatorInputs, EnsembleOperatorOutputs]):
    """Operator that executes parallel calls to multiple LMModules concurrently."""

    signature: Signature = Signature(
        input_model=EnsembleOperatorInputs, output_model=EnsembleOperatorOutputs
    )
    lm_modules: List[LMModule]

    def __init__(self, *, lm_modules: List[LMModule]) -> None:
        self.lm_modules = lm_modules

    def forward(self, *, inputs: EnsembleOperatorInputs) -> EnsembleOperatorOutputs:
        rendered_prompt: str = self.signature.render_prompt(inputs=inputs)
        responses: List[str] = [lm(prompt=rendered_prompt) for lm in self.lm_modules]
        return {"responses": responses}
