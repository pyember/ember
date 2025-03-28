"""Minimal Example

This module demonstrates the simplest possible Ember operator.

This example illustrates how the Ember `Operator` construct provides a minimal
foundation that can be used for language-model transformations or to wrap
arbitrary function logic.

To run:
    poetry run python src/ember/examples/basic/minimal_example.py
"""

from typing import Any, Dict, List, Optional, Type

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field


class MinimalInput(EmberModel):
    """Input model for MinimalOperator.

    Attributes:
        value: The integer value to be processed
        options: Optional configuration parameters
    """

    value: int = Field(description="The integer value to be processed")
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional configuration parameters"
    )


class MinimalOutput(EmberModel):
    """Output model for MinimalOperator.

    Attributes:
        value: The processed value
        steps: List of processing steps applied
    """

    value: int = Field(description="The processed value")
    steps: List[str] = Field(
        default_factory=list, description="List of processing steps applied"
    )

    def summarize(self) -> str:
        """Get a summary of the processing result.

        Returns:
            A formatted summary string
        """
        step_text = "\n".join(f"- {step}" for step in self.steps)
        return f"Result: {self.value}\nSteps applied:\n{step_text}"


class MinimalSpecification(Specification):
    """Specification for MinimalOperator.

    Defines the input/output contract that the operator fulfills.
    """

    input_model: Type[EmberModel] = MinimalInput
    structured_output: Type[EmberModel] = MinimalOutput


class MinimalOperator(Operator[MinimalInput, MinimalOutput]):
    """A minimal operator that performs configurable numeric operations.

    This operator demonstrates the clean patterns for creating Ember operators
    with proper typing, immutability, and clean interfaces.

    Attributes:
        specification: The operator's input/output contract
        increment: The value to add to the input
        multiplier: The value to multiply by after incrementing
    """

    # Class-level specification (immutable)
    specification: Specification = MinimalSpecification()

    # Instance attributes with type hints
    increment: int
    multiplier: int

    def __init__(self, *, increment: int = 1, multiplier: int = 1) -> None:
        """Initialize with transformation parameters.

        Args:
            increment: The value to add to the input
            multiplier: The value to multiply by after incrementing
        """
        self.increment = increment
        self.multiplier = multiplier

    def forward(self, *, inputs: MinimalInput) -> MinimalOutput:
        """Process the input value with configured transformations.

        Args:
            inputs: Validated input data containing the value to process

        Returns:
            Structured output containing the processed value and steps
        """
        # Track processing steps
        steps = []

        # Apply increment
        result = inputs.value + self.increment
        steps.append(f"Added {self.increment} to {inputs.value} = {result}")

        # Apply multiplier
        if self.multiplier != 1:
            prev = result
            result *= self.multiplier
            steps.append(f"Multiplied {prev} by {self.multiplier} = {result}")

        # Apply any custom options if provided
        if inputs.options:
            if "square" in inputs.options and inputs.options["square"]:
                prev = result
                result = result**2
                steps.append(f"Squared {prev} = {result}")

            if "add" in inputs.options:
                value = inputs.options["add"]
                prev = result
                result += value
                steps.append(f"Added {value} to {prev} = {result}")

        # Return structured output
        return MinimalOutput(value=result, steps=steps)


def main() -> None:
    """Run a simple demonstration of the MinimalOperator."""
    print("\n=== Minimal Operator Example ===\n")

    # Create the operator
    op = MinimalOperator(increment=5, multiplier=2)

    # Create input with basic value
    basic_input = MinimalInput(value=10)

    # Process with only the basic configuration
    basic_result = op(inputs=basic_input)

    print("Basic Example:")
    print(f"  Input: {basic_input.value}")
    print(f"  Result: {basic_result.value}")
    print("  Steps:")
    for step in basic_result.steps:
        print(f"    {step}")

    # Create input with advanced options
    advanced_input = MinimalInput(value=7, options={"square": True, "add": 3})

    # Process with advanced options
    advanced_result = op(inputs=advanced_input)

    print("\nAdvanced Example:")
    print(f"  Input: {advanced_input.value} with options {advanced_input.options}")
    print(f"  Result: {advanced_result.value}")
    print("  Steps:")
    for step in advanced_result.steps:
        print(f"    {step}")

    # Alternative invocation patterns
    print("\nAlternative Invocation Patterns:")

    # Using dict input
    dict_result = op(inputs={"value": 3})
    print(f"  Dict input result: {dict_result.value}")

    # Using keyword arguments
    kwargs_result = op(value=4)
    print(f"  Keyword args result: {kwargs_result.value}")

    print("\nSummary using model method:")
    print(advanced_result.summarize())
    print()


if __name__ == "__main__":
    main()
