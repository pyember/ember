from typing import Callable, Dict, Any, List
from random import sample

from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core import non
from src.ember.core.registry.operator.base._module import ember_field
from src.ember.core.registry.model.model_modules.lm import LMModule


def usage_example() -> None:
    """Demonstrates usage of MultiPrefixEnsembleOperator with distinct prefixes for each language model.

    This function creates a MultiPrefixEnsembleOperator with example prefixes and mock language model modules,
    constructs sample input, executes the operator, and prints the aggregated responses.

    Returns:
        None.
    """
    # Define example prefixes and mock language model functions.
    example_prefixes: List[str] = ["PrefixA", "PrefixB", "PrefixC"]
    mock_lm_modules: List[Callable[[str], str]] = [
        lambda prompt: f"[MockResponse] for prompt: {prompt}",
        lambda prompt: f"[MockResponse] for prompt: {prompt}",
        lambda prompt: f"[MockResponse] for prompt: {prompt}",
    ]

    # Instantiate the operator with named parameters.
    operator: MultiPrefixEnsembleOperator = MultiPrefixEnsembleOperator(
        model_name="anthropic:claude-3-opus",
        prefixes=example_prefixes,
        name="MultiPrefixEnsembleExample",
        num_units=3,
        temperature=0.5,
    )

    # Create input data.
    inputs: MultiPrefixOperatorInputs = MultiPrefixOperatorInputs(
        query="What's the best approach?"
    )

    # Execute the operator's forward method using named invocation.
    result: Dict[str, Any] = operator.forward(inputs=inputs)
    print("Usage Example Result:", result)


class MultiPrefixOperatorInputs(BaseModel):
    """Input model for MultiPrefixEnsembleOperator.

    Attributes:
        query: The query string to be processed by the operator.
    """

    query: str


class MultiPrefixEnsembleOperator(Operator[MultiPrefixOperatorInputs, Dict[str, Any]]):
    """Operator that applies different prefixes using UniformEnsemble."""

    signature: Signature = Signature(
        required_inputs=["query"],
        input_model=MultiPrefixOperatorInputs,
    )
    lm_modules: List[LMModule]

    def __init__(
        self,
        lm_modules: List[LMModule],
        prefixes: List[str],
        name: str = "MultiPrefixEnsemble",
    ) -> None:
        """Initializes a MultiPrefixEnsembleOperator instance.

        Args:
            lm_modules: A list of language model callables.
            prefixes: A list of prefix strings to be used for each LM call.
            name: The name identifier for this operator instance.
        """
        self.prefixes: List[str] = prefixes
        self.lm_modules: List[LMModule] = lm_modules

    def forward(self, inputs: MultiPrefixOperatorInputs) -> Dict[str, Any]:
        chosen_prefixes = sample(self.prefixes, len(self.lm_modules))

        responses: List[str] = []
        for prefix, lm in zip(chosen_prefixes, self.lm_modules):
            prompt = f"{prefix}\n{inputs.query}"
            response = lm(prompt=prompt)
            responses.append(response)
        return {"responses": responses}


def main() -> None:
    """Main entry point that runs the usage example.

    Returns:
        None.
    """
    usage_example()


if __name__ == "__main__":
    main()
