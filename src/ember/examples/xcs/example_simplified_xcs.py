"""
Example demonstrating the simplified XCS import structure.

This example shows how to use the new top-level imports for XCS functionality
with the ember.api package.

To run:
    poetry run python src/ember/examples/example_simplified_xcs.py
"""

from typing import ClassVar, Optional, Type

from ember.api.xcs import jit, pmap, vmap
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel

# Import the API for advanced configuration


# Create input/output models for our operators
class QueryInput(EmberModel):
    query: str


class QueryOutput(EmberModel):
    result: str


# Define specifications for our operators
class SimpleOperatorSpec(Specification[QueryInput, QueryOutput]):
    input_model: Optional[Type[QueryInput]] = QueryInput
    structured_output: Optional[Type[QueryOutput]] = QueryOutput


# Create a simple operator
@jit  # Simple JIT usage
class SimpleOperator(Operator[QueryInput, QueryOutput]):
    # Define the specification
    specification: ClassVar[Specification] = SimpleOperatorSpec()

    def forward(self, *, inputs: QueryInput) -> QueryOutput:
        return QueryOutput(result=inputs.query.upper())


# Define specification for advanced operator
class AdvancedOperatorSpec(Specification[QueryInput, QueryOutput]):
    input_model: Optional[Type[QueryInput]] = QueryInput
    structured_output: Optional[Type[QueryOutput]] = QueryOutput


# Use advanced JIT options
@jit(sample_input={"query": "precompile"})
class AdvancedOperator(Operator[QueryInput, QueryOutput]):
    # Define the specification
    specification: ClassVar[Specification] = AdvancedOperatorSpec()

    def forward(self, *, inputs: QueryInput) -> QueryOutput:
        return QueryOutput(result=inputs.query + "!")


def main():
    """Run the example demonstrating simplified XCS imports."""
    print("\n=== Simplified XCS Import Example ===\n")

    # Create and use the operators
    simple_op = SimpleOperator()
    advanced_op = AdvancedOperator()

    # Demonstrate the operators in action
    print("Simple Operator Demo:")
    result1 = simple_op(inputs={"query": "hello world"})
    print("  Input: 'hello world'")
    print(f"  Output: '{result1['result']}'")  # Should be "HELLO WORLD"

    print("\nAdvanced Operator Demo:")
    result2 = advanced_op(inputs={"query": "precompiled input"})
    print("  Input: 'precompiled input'")
    print(f"  Output: '{result2['result']}'")  # Should be "precompiled input!"

    # Vectorization example
    def process_item(inputs):
        # Process the input and return a dictionary
        return {"result": inputs["values"] * 2}

    # Vectorize the function
    print("\nVectorization Example:")
    batch_process = vmap(process_item)
    input_values = [1, 2, 3]
    # Note: vmap expects a dict with keyword 'inputs'
    batch_result = batch_process(inputs={"values": input_values})
    print(f"  Inputs: {input_values}")
    print(f"  Vectorized Output: {batch_result}")  # Should be {"result": [2, 4, 6]}

    # Parallelize the function
    print("\nParallelization Example:")
    parallel_process = pmap(process_item)
    print("  The pmap decorator enables parallel processing across multiple cores")
    print("  Usage: parallel_process(inputs={'values': [1, 2, 3]})")

    # Show autograph example
    print("\nAutograph Example:")
    print("  The autograph decorator captures function calls as a computational graph")
    print("  @autograph")
    print("  def my_function(x):")
    print("      return process1(process2(x))")

    # Show execution options
    print("\nExecution Options Example:")
    print("  with execution_options(scheduler='parallel'):")
    print("      result = my_complex_operation(data)")

    print("\nXCS API import example complete!")
    print(
        "These APIs provide a simple, intuitive interface to Ember's execution framework."
    )


if __name__ == "__main__":
    main()
