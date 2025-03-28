"""Minimal Operator Example

This script demonstrates the recommended pattern for creating and using Ember operators.
It shows how to create, configure, and execute operators with proper typing and error handling.

To run:
    uv run python src/ember/examples/basic/minimal_operator_example.py
"""

# Import the minimal example
from ember.examples.basic.minimal_example import MinimalInput, MinimalOperator, main

if __name__ == "__main__":
    # Run the example from the minimal_example module
    main()

    # Additional demonstration of composition
    print("\n=== Operator Composition Example ===\n")

    # Create operators with different configurations
    op1 = MinimalOperator(increment=2, multiplier=3)
    op2 = MinimalOperator(increment=10, multiplier=1)

    # Create an input
    input_data = MinimalInput(value=5)

    # Process through the first operator
    intermediate = op1(inputs=input_data)
    print(f"After first operator: {intermediate.value}")

    # Process through the second operator
    final = op2(inputs=MinimalInput(value=intermediate.value))
    print(f"After second operator: {final.value}")

    # Show all processing steps
    print("\nAll processing steps:")
    for step in intermediate.steps:
        print(f"  {step}")
    for step in final.steps:
        print(f"  {step}")
    print()
