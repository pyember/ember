"""Simplified Automatic Graph Building Example.

This example demonstrates the enhanced JIT API with automatic graph building
without requiring LLM API calls. It shows how the @jit decorator can automatically
optimize operator execution by tracing execution patterns and building optimized
execution graphs.

This example highlights the trace-based JIT approach using the @jit decorator,
which is one of three complementary approaches in Ember's JIT system (the others
being structural_jit for structure-based analysis and autograph for manual graph
construction).

For a comprehensive explanation of the relationship between these approaches,
see docs/xcs/JIT_OVERVIEW.md.

To run:
    uv run python src/ember/examples/xcs/auto_graph_simplified.py
"""

import logging
import time
from typing import ClassVar, Type

from pydantic import Field

from ember.api.xcs import execution_options, jit
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel

###############################################################################
# Input/Output Models
###############################################################################


class AdditionInput(EmberModel):
    """Input model for math operations.

    Attributes:
        value: The integer value to be processed.
    """

    value: int = Field(description="The integer value to be processed")


class AdditionOutput(EmberModel):
    """Output model for math operations.

    Attributes:
        value: The processed integer value.
    """

    value: int = Field(description="The processed integer value")


class AdditionSpec(Specification):
    """Specification for math operation operators."""

    input_model: Type[EmberModel] = AdditionInput
    structured_output: Type[EmberModel] = AdditionOutput


###############################################################################
# JIT-Decorated Operators
###############################################################################


@jit()
class AddTenOperator(Operator[AdditionInput, AdditionOutput]):
    """Operator that adds 10 to the input."""

    specification: ClassVar[Specification] = AdditionSpec()

    def forward(self, *, inputs: AdditionInput) -> AdditionOutput:
        """Add 10 to the input value.

        Args:
            inputs: The input containing the value to increment by 10.

        Returns:
            Output with the value incremented by 10.
        """
        time.sleep(0.1)  # Simulate processing time
        return AdditionOutput(value=inputs.value + 10)


@jit()
class MultiplyByTwoOperator(Operator[AdditionInput, AdditionOutput]):
    """Operator that multiplies the input by 2."""

    specification: ClassVar[Specification] = AdditionSpec()

    def forward(self, *, inputs: AdditionInput) -> AdditionOutput:
        """Multiply the input value by 2.

        Args:
            inputs: The input containing the value to multiply by 2.

        Returns:
            Output with the value multiplied by 2.
        """
        time.sleep(0.1)  # Simulate processing time
        return AdditionOutput(value=inputs.value * 2)


###############################################################################
# Pipeline Class
###############################################################################


@jit()
class MathPipeline(Operator[AdditionInput, AdditionOutput]):
    """Pipeline that demonstrates automatic graph building."""

    specification: ClassVar[Specification] = AdditionSpec()

    # Define instance attributes with type hints
    add_ten: AddTenOperator
    multiply: MultiplyByTwoOperator

    def __init__(self) -> None:
        """Initialize with add_ten and multiply operators."""
        self.add_ten = AddTenOperator()
        self.multiply = MultiplyByTwoOperator()

    def forward(self, *, inputs: AdditionInput) -> AdditionOutput:
        """Execute the pipeline with automatic graph building.

        Args:
            inputs: The input containing the initial value for processing.

        Returns:
            Output with the value after both operations are applied.
        """
        # First add 10
        add_result = self.add_ten(inputs=inputs)

        # Then multiply by 2
        final_result = self.multiply(inputs=add_result)

        return final_result


def main() -> None:
    """Run demonstration of automatic graph building."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Simplified Automatic Graph Building Example ===\n")

    # Create the pipeline
    pipeline = MathPipeline()

    # Test with different inputs
    inputs = [
        AdditionInput(value=5),  # Should result in (5+10)*2 = 30
        AdditionInput(value=10),  # Should result in (10+10)*2 = 40
        AdditionInput(value=15),  # Should result in (15+10)*2 = 50
    ]

    # First run - trace and build graph
    print("First run - expect graph building overhead:")
    for i, input_data in enumerate(inputs):
        start_time = time.time()
        result = pipeline(inputs=input_data)
        elapsed = time.time() - start_time
        print(f"\nInput {i+1}: {input_data}")
        print(f"Result: {result}")
        print(f"Value from result: {result.value}")
        # Show computation steps
        print(
            f"Computation: {input_data.value} + 10 = {input_data.value + 10}, then × 2 = {(input_data.value + 10) * 2}"
        )
        print(f"Time: {elapsed:.4f}s")

    # Second run - should use cached graph
    print("\nRepeat first input to demonstrate cached execution:")
    start_time = time.time()
    result = pipeline(inputs=inputs[0])
    elapsed = time.time() - start_time
    print(f"Result: {result}")
    print(f"Value from result: {result.value}")
    print(f"Time: {elapsed:.4f}s")
    print("Note: Second run is typically faster due to cached execution")

    # With execution options
    print("\nUsing execution_options to control execution:")
    with execution_options(scheduler="sequential"):
        start_time = time.time()
        result = pipeline(
            inputs=AdditionInput(value=20)
        )  # Should result in (20+10)*2 = 60
        elapsed = time.time() - start_time
        print(f"Result: {result}")
        print(f"Value from result: {result.value}")
        # Show computation steps
        print("Computation: 20 + 10 = 30, then × 2 = 60")
        print(f"Time: {elapsed:.4f}s (sequential execution)")


if __name__ == "__main__":
    main()
