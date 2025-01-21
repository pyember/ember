from pydantic import BaseModel
from typing import Dict, Any, List
from random import sample

from src.avior.registry.operator.operator_base import Operator, OperatorType, OperatorMetadata
from src.avior.registry.prompt_signature.signatures import Signature

def usage_example():
    """
    Demonstrates how to use MultiPrefixEnsembleOperator with different prefixes per LM.
    """
    # Example prefixes and LM modules
    example_prefixes = ["PrefixA", "PrefixB", "PrefixC"] 
    mock_lm_modules = [
        lambda prompt: f"[MockResponse] with prompt: {prompt}",
        lambda prompt: f"[MockResponse] with prompt: {prompt}",
        lambda prompt: f"[MockResponse] with prompt: {prompt}",
    ]

    # Instantiate the operator
    operator = MultiPrefixEnsembleOperator(
        lm_modules=mock_lm_modules,
        prefixes=example_prefixes,
        name="MultiPrefixEnsembleExample"
    )

    # Create inputs
    inputs = MultiPrefixOperatorInputs(query="What's the best approach?")

    # Execute
    result = operator.forward(inputs)
    print("Usage Example Result:", result)


class MultiPrefixOperatorInputs(BaseModel):
    query: str


class MultiPrefixEnsembleOperator(Operator[MultiPrefixOperatorInputs, Dict[str, Any]]):
    metadata = OperatorMetadata(
        code="MP_ENSEMBLE",
        description="Ensemble with different prefix per LM call",
        operator_type=OperatorType.FAN_OUT,
        signature=Signature(
            required_inputs=["query"],
            input_model=MultiPrefixOperatorInputs,
        ),
    )

    def __init__(self, lm_modules, prefixes: List[str], name="MultiPrefixEnsemble"):
        super().__init__(name=name, lm_modules=lm_modules)
        self.prefixes = prefixes  # e.g. ["PREFIX_1", "PREFIX_2", ...]

    def forward(self, inputs: MultiPrefixOperatorInputs) -> Dict[str, Any]:
        chosen_prefixes = sample(self.prefixes, len(self.lm_modules))
        responses = []
        for lm, prefix in zip(self.lm_modules, chosen_prefixes):
            prompt = prefix + "\n" + inputs.query
            resp = lm(prompt)
            responses.append(resp)
        return {"responses": responses}

def main():
    usage_example()

if __name__ == "__main__":
    main()