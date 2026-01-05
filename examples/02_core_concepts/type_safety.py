"""Leverage Python's type system for safer Ember code.

This example demonstrates:
- Type annotations for operators
- Generic operators with TypeVar
- Runtime type checking with Pydantic
- Static analysis compatibility

Run with:
    python examples/02_core_concepts/type_safety.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ember.api import op


# =============================================================================
# Part 1: Basic Type Annotations
# =============================================================================

@op
def transform_text(text: str, uppercase: bool = False) -> str:
    """Transform text with typed parameters.

    Type hints enable:
    - IDE autocomplete and error detection
    - Documentation generation
    - Runtime validation (with Pydantic)
    """
    result = text.strip()
    if uppercase:
        result = result.upper()
    return result


@op
def calculate_stats(numbers: list[float]) -> dict[str, float]:
    """Calculate statistics with precise type hints."""
    if not numbers:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0.0}

    return {
        "min": min(numbers),
        "max": max(numbers),
        "mean": sum(numbers) / len(numbers),
        "count": float(len(numbers)),
    }


# =============================================================================
# Part 2: Dataclass Types
# =============================================================================

@dataclass
class TextInput:
    """Strongly typed input for text processing."""

    content: str
    language: str = "en"
    max_length: int | None = None


@dataclass
class TextOutput:
    """Strongly typed output from text processing."""

    processed: str
    original_length: int
    truncated: bool


@op
def process_typed_text(input_data: TextInput) -> TextOutput:
    """Process text using dataclass types.

    Dataclasses provide:
    - Clear structure
    - Default values
    - Automatic __repr__ for debugging
    """
    content = input_data.content
    original_length = len(content)
    truncated = False

    if input_data.max_length and len(content) > input_data.max_length:
        content = content[: input_data.max_length] + "..."
        truncated = True

    return TextOutput(
        processed=content,
        original_length=original_length,
        truncated=truncated,
    )


# =============================================================================
# Part 3: Generic Operators
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")


class Pipeline(Generic[T, R]):
    """A typed pipeline that transforms T to R.

    Demonstrates generic typing for reusable patterns.
    """

    def __init__(self, name: str):
        self.name = name
        self._steps: list = []

    def add_step(self, step) -> "Pipeline[T, R]":
        """Add a processing step."""
        self._steps.append(step)
        return self

    def run(self, input_data: T) -> R:
        """Execute the pipeline."""
        result = input_data
        for step in self._steps:
            result = step(result)
        return result  # type: ignore[return-value]


@op
def double(x: int) -> int:
    """Double an integer."""
    return x * 2


@op
def to_string(x: int) -> str:
    """Convert integer to string."""
    return f"Result: {x}"


# =============================================================================
# Part 4: Type Guards and Narrowing
# =============================================================================

def is_valid_number(value: str | int | float) -> bool:
    """Type guard to check if value is numeric."""
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


@op
def safe_multiply(a: str | int | float, b: str | int | float) -> float | None:
    """Multiply values with type checking.

    Demonstrates defensive type handling.
    """
    # Coerce to float
    def to_float(v: str | int | float) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None

    a_float = to_float(a)
    b_float = to_float(b)

    if a_float is None or b_float is None:
        return None

    return a_float * b_float


def main() -> None:
    """Demonstrate type safety patterns."""
    print("Type Safety in Ember")
    print("=" * 50)
    print()

    # Part 1: Basic types
    print("Part 1: Basic Type Annotations")
    print("-" * 50)

    result = transform_text("  hello world  ", uppercase=True)
    print(f"Transformed: {result}")

    stats = calculate_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Stats: {stats}")
    print()

    # Part 2: Dataclasses
    print("Part 2: Dataclass Types")
    print("-" * 50)

    input_data = TextInput(
        content="This is a long piece of text that might need truncation.",
        language="en",
        max_length=20,
    )
    output = process_typed_text(input_data)
    print(f"Input:  {input_data}")
    print(f"Output: {output}")
    print()

    # Part 3: Generic pipeline
    print("Part 3: Generic Typed Pipeline")
    print("-" * 50)

    pipeline: Pipeline[int, str] = Pipeline("int_to_str")
    pipeline.add_step(double).add_step(double).add_step(to_string)

    result = pipeline.run(5)
    print(f"Pipeline result: {result}")
    print()

    # Part 4: Safe operations
    print("Part 4: Type-Safe Operations")
    print("-" * 50)

    test_cases = [
        (5, 3),
        ("2.5", 4),
        ("hello", 3),
        (10, "invalid"),
    ]
    for a, b in test_cases:
        result = safe_multiply(a, b)
        status = "OK" if result is not None else "Failed"
        print(f"  {a!r} * {b!r} = {result} ({status})")
    print()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Type hints enable IDE support and static analysis")
    print("2. Dataclasses provide structured, documented types")
    print("3. Generics allow reusable typed patterns")
    print("4. Type guards enable safe narrowing")
    print()
    print("Next: Explore examples/03_simplified_apis/ for ergonomic patterns")


if __name__ == "__main__":
    main()
