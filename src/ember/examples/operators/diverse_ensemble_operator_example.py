"""Diverse Ensemble Operator Example

This example demonstrates using the MultiPrefixEnsembleOperator with distinct prefixes
for each language model to generate diverse responses to a query.

To run:
    uv run python src/ember/examples/diverse_ensemble_operator_example.py
"""

from random import sample
from typing import ClassVar, List, Type

from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field


def usage_example() -> None:
    """Demonstrates usage of MultiPrefixEnsembleOperator with distinct prefixes for each language model.

    This function creates a MultiPrefixEnsembleOperator with example prefixes and language model modules,
    constructs sample input, executes the operator, and prints the aggregated responses.

    Returns:
        None.
    """

    # Define example prefixes to guide different response styles
    example_prefixes: List[str] = [
        "Analyze this from a scientific perspective:",
        "Consider this from a philosophical angle:",
        "Provide a practical approach to:",
    ]

    # Create LM modules with different models for diversity
    lm_modules = [
        LMModule(
            config=LMModuleConfig(
                model_name="anthropic:claude-3-opus", temperature=0.5, max_tokens=256
            )
        ),
        LMModule(
            config=LMModuleConfig(
                model_name="openai:gpt-4", temperature=0.7, max_tokens=256
            )
        ),
        LMModule(
            config=LMModuleConfig(
                model_name="anthropic:claude-3-haiku", temperature=0.3, max_tokens=256
            )
        ),
    ]

    # Instantiate the operator with named parameters.
    operator: MultiPrefixEnsembleOperator = MultiPrefixEnsembleOperator(
        lm_modules=lm_modules,
        prefixes=example_prefixes,
        name="MultiPrefixEnsembleExample",
    )

    # Create input data with a more substantive query
    inputs: MultiPrefixOperatorInputs = MultiPrefixOperatorInputs(
        query="How can we effectively combat climate change while balancing economic needs?"
    )

    # Execute the operator using __call__ with named parameters
    result = operator(inputs=inputs)

    # Display structured results
    print(f'Original query: "{inputs.query}"')
    print(f"\nNumber of responses: {len(result.responses)}")

    # Display each response with its corresponding prefix
    for i, (prefix, response) in enumerate(zip(example_prefixes, result.responses), 1):
        # Show the prefix and a truncated response for readability
        truncated = response[:100] + "..." if len(response) > 100 else response
        print(f"\nResponse {i}:")
        print(f'  Prefix: "{prefix}"')
        print(f'  Response: "{truncated}"')

    print(
        "\nNote: In a real application, these responses would be further processed or aggregated."
    )


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
    structured_output: Type[EmberModel] = MultiPrefixOperatorOutputs


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

            # Get text from response
            text = response if isinstance(response, str) else str(response)

            # Ensure we have a valid string
            text = text.strip() if text else ""

            responses.append(text)

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
