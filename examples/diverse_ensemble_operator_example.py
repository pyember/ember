from typing import ClassVar, Dict, List, Optional, Type
from random import sample

from pydantic import Field

from ember.core.app_context import get_ember_context
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.prompt_specification.specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.core import non
from ember.core.registry.operator.base._module import ember_field
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.utils import extract_value


def usage_example() -> None:
    """Demonstrates usage of MultiPrefixEnsembleOperator with distinct prefixes for each language model.

    This function creates a MultiPrefixEnsembleOperator with example prefixes and language model modules,
    constructs sample input, executes the operator, and prints the aggregated responses.

    Returns:
        None.
    """
    # Initialize ember context
    context = get_ember_context()

    # Define example prefixes
    example_prefixes: List[str] = ["PrefixA", "PrefixB", "PrefixC"]

    # Create LM modules through the context
    lm_modules = [
        LMModule(
            config=LMModuleConfig(
                model_name="anthropic:claude-3-opus", temperature=0.5, max_tokens=256
            )
        )
        for _ in range(3)
    ]

    # Instantiate the operator with named parameters.
    operator: MultiPrefixEnsembleOperator = MultiPrefixEnsembleOperator(
        lm_modules=lm_modules,
        prefixes=example_prefixes,
        name="MultiPrefixEnsembleExample",
    )

    # Create input data.
    inputs: MultiPrefixOperatorInputs = MultiPrefixOperatorInputs(
        query="What's the best approach?"
    )

    # Execute the operator using __call__ with named parameters
    result = operator(inputs=inputs)

    # Display structured results
    print(f"Number of responses: {len(result.responses)}")
    for i, response in enumerate(result.responses, 1):
        print(f"Response {i}: {response[:50]}..." if len(response) > 50 else response)


class MultiPrefixOperatorInputs(EmberModel):
    """Input model for MultiPrefixEnsembleOperator.

    Attributes:
        query: The query string to be processed by the operator.
    """

    query: str = Field(description="The query to be processed by multiple LM modules")


class MultiPrefixOperatorOutputs(EmberModel):
    """Output model for MultiPrefixEnsembleOperator.

    Attributes:
        responses: The list of responses from different LM modules.
    """

    responses: List[str] = Field(description="Responses from different LM modules")


class MultiPrefixEnsembleSpecification(Specification):
    """Specification for MultiPrefixEnsembleOperator."""

    input_model: Type[EmberModel] = MultiPrefixOperatorInputs
    output_model: Type[EmberModel] = MultiPrefixOperatorOutputs


class MultiPrefixEnsembleOperator(
    Operator[MultiPrefixOperatorInputs, MultiPrefixOperatorOutputs]
):
    """Operator that applies different prefixes using multiple LM modules.

    This operator randomly selects prefixes from a predefined list and applies them
    to the user query before sending to different language model modules.
    """

    specification: ClassVar[Specification] = MultiPrefixEnsembleSpecification()
    lm_modules: List[LMModule]
    prefixes: List[str]

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
        self.prefixes = prefixes
        self.lm_modules = lm_modules

    def forward(
        self, *, inputs: MultiPrefixOperatorInputs
    ) -> MultiPrefixOperatorOutputs:
        """Apply different prefixes to the query and process through LM modules.

        Args:
            inputs: Validated input data containing the query.

        Returns:
            Structured output containing responses from all LM modules.
        """
        # Randomly select prefixes to match the number of LM modules
        chosen_prefixes = sample(self.prefixes, len(self.lm_modules))

        # Process each query with a different prefix through its LM module
        responses = []
        for prefix, lm in zip(chosen_prefixes, self.lm_modules):
            # Generate prompt with prefix
            prompt = f"{prefix}\n{inputs.query}"

            # Call LM module
            response = lm(prompt=prompt)

            # Guard against None responses
            if response is None:
                response = ""

            responses.append(response)

        # Return structured output
        return MultiPrefixOperatorOutputs(responses=responses)


def main() -> None:
    """Main entry point that runs the usage example.

    Returns:
        None.
    """
    usage_example()


if __name__ == "__main__":
    main()
