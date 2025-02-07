from typing import Callable, Dict, Any, List
from random import sample

from pydantic import BaseModel

from ember.registry.operator.operator_base import (
    Operator,
    OperatorType,
    OperatorMetadata,
)
from ember.registry.prompt_signature.signatures import Signature


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
        lm_modules=mock_lm_modules,
        prefixes=example_prefixes,
        name="MultiPrefixEnsembleExample",
    )

    # Create input data.
    inputs: MultiPrefixOperatorInputs = MultiPrefixOperatorInputs(query="What's the best approach?")
    
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
    """Operator that applies different prefixes for each language model call in an ensemble.

    This operator randomly selects a prefix for each language model, concatenates it with the provided query,
    and aggregates the responses.

    Attributes:
        metadata: Operator metadata describing the operator's code, description, type, and input signature.
        prefixes: List of prefix strings used to modify the query for each language model.
        lm_modules: List of callables representing language model modules. Each callable takes a prompt string
            and returns a response string.
    """
    metadata: OperatorMetadata = OperatorMetadata(
        code="MP_ENSEMBLE",
        description="Ensemble with distinct prefix per LM invocation.",
        operator_type=OperatorType.FAN_OUT,
        signature=Signature(
            required_inputs=["query"],
            input_model=MultiPrefixOperatorInputs,
        ),
    )

    def __init__(
        self,
        lm_modules: List[Callable[[str], str]],
        prefixes: List[str],
        name: str = "MultiPrefixEnsemble",
    ) -> None:
        """Initializes a MultiPrefixEnsembleOperator instance.

        Args:
            lm_modules: A list of language model callables.
            prefixes: A list of prefix strings to be used for each LM call.
            name: The name identifier for this operator instance.
        """
        super().__init__(name=name, lm_modules=lm_modules)
        self.prefixes: List[str] = prefixes

    def forward(self, inputs: MultiPrefixOperatorInputs) -> Dict[str, Any]:
        """Executes the operator by prepending randomly selected prefixes to the input query.

        For each language model module, a prefix is chosen at random, concatenated with the query separated by
        a newline, and passed to the language model module. The responses are aggregated into a dictionary.

        Args:
            inputs: An instance of MultiPrefixOperatorInputs containing the query.

        Returns:
            A dictionary with a single key 'responses' mapping to a list of response strings from the LM modules.
        """
        chosen_prefixes: List[str] = sample(self.prefixes, len(self.lm_modules))
        responses: List[str] = []
        for lm, prefix in zip(self.lm_modules, chosen_prefixes):
            prompt: str = f"{prefix}\n{inputs.query}"
            response: str = lm(prompt=prompt)
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
