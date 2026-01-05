"""Verify Ember installation and demonstrate basic concepts.

This example runs without API keys and shows:
- Package import verification
- Version information
- Basic operator creation with @op decorator
- Simple function composition

Run with:
    python examples/01_getting_started/hello_world.py
"""

from __future__ import annotations

import ember
from ember.api import op


@op
def greet(name: str) -> str:
    """Create a personalized greeting."""
    return f"Hello, {name}! Welcome to Ember."


@op
def format_greeting(greeting: str, style: str = "friendly") -> dict:
    """Format a greeting with metadata."""
    styles = {
        "friendly": greeting,
        "formal": greeting.replace("Hello", "Greetings"),
        "casual": greeting.replace("Hello", "Hey"),
    }
    return {
        "message": styles.get(style, greeting),
        "style": style,
        "length": len(greeting),
    }


def main() -> None:
    """Demonstrate basic Ember functionality."""
    # Verify installation
    print("Ember Installation Check")
    print("=" * 40)
    print(f"Version: {ember.__version__}")
    print()

    # Basic operator usage
    print("Basic Operator Usage")
    print("-" * 40)

    greeting = greet("Developer")
    print(f"Simple greeting: {greeting}")

    formatted = format_greeting(greeting, style="formal")
    print(f"Formatted: {formatted}")
    print()

    # Composition example
    print("Function Composition")
    print("-" * 40)

    # Operators compose naturally as Python functions
    result = format_greeting(greet("World"), style="casual")
    print(f"Composed result: {result}")
    print()

    print("Installation verified. Ember is ready to use.")
    print()
    print("Next steps:")
    print("  1. Run 'ember setup' to configure API credentials")
    print("  2. Try examples/01_getting_started/first_model_call.py")


if __name__ == "__main__":
    main()
