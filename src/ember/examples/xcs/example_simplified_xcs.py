"""
Example demonstrating the simplified XCS import structure.

This example shows how to use the new top-level imports for XCS functionality
with the ember.api package.

To run:
    poetry run python src/ember/examples/example_simplified_xcs.py
"""

from ember.api.xcs import jit, pmap, vmap
from ember.core.registry.operator.base.operator_base import Operator
from ember.xcs.engine.execution_options import execution_options
from ember.xcs.tracer.autograph import autograph

# Import the API for advanced configuration
from ember.xcs.tracer.tracer_decorator import JITOptions


# Create a simple operator
@jit  # Simple JIT usage
class SimpleOperator(Operator):
    def forward(self, *, inputs):
        return {"result": inputs["query"].upper()}


# Use advanced JIT options
@jit(options=JITOptions(sample_input={"query": "precompile"}))
class AdvancedOperator(Operator):
    def forward(self, *, inputs):
        return {"result": inputs["query"] + "!"}


def main():
    """Run the example demonstrating simplified XCS imports."""
    print("\n=== Simplified XCS Import Example ===\n")

    # Create and use the operators
    simple_op = SimpleOperator()
    advanced_op = AdvancedOperator()

    # Demonstrate the operators in action
    print("Simple Operator Demo:")
    result1 = simple_op(inputs={"query": "hello world"})
    print(f"  Input: 'hello world'")
    print(f"  Output: '{result1['result']}'")  # Should be "HELLO WORLD"

    print("\nAdvanced Operator Demo:")
    result2 = advanced_op(inputs={"query": "precompiled input"})
    print(f"  Input: 'precompiled input'")
    print(f"  Output: '{result2['result']}'")  # Should be "precompiled input!"

    # Vectorization example
    def process_item(item):
        return item * 2

    # Vectorize the function
    print("\nVectorization Example:")
    batch_process = vmap(process_item)
    inputs = [1, 2, 3]
    batch_result = batch_process(inputs)
    print(f"  Inputs: {inputs}")
    print(f"  Vectorized Output: {batch_result}")  # Should be [2, 4, 6]

    # Parallelize the function
    print("\nParallelization Example:")
    parallel_process = pmap(process_item)
    print(f"  The pmap decorator enables parallel processing across multiple cores")
    print(f"  Usage: parallel_process([1, 2, 3])")

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
