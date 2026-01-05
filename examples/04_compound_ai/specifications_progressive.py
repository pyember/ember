"""Build specifications progressively for complex AI systems.

This example demonstrates:
- Starting with minimal specs
- Adding constraints incrementally
- Composing specifications
- Validation at each level

Run with:
    python examples/04_compound_ai/specifications_progressive.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ember.api import op


# =============================================================================
# Part 1: Level 1 - No Specification (Duck Typing)
# =============================================================================

@op
def process_any(data: Any) -> Any:
    """Process any input, return any output.

    Level 1: Maximum flexibility, minimum safety.
    Works for prototyping, not recommended for production.
    """
    if isinstance(data, str):
        return {"type": "string", "value": data.upper()}
    if isinstance(data, dict):
        return {"type": "dict", "keys": list(data.keys())}
    return {"type": str(type(data).__name__), "value": str(data)}


# =============================================================================
# Part 2: Level 2 - Type Hints
# =============================================================================

@op
def process_typed(text: str, options: dict | None = None) -> dict[str, Any]:
    """Process with type hints.

    Level 2: Static type checking, IDE support.
    """
    opts = options or {}
    return {
        "text": text,
        "length": len(text),
        "uppercase": opts.get("uppercase", False),
        "result": text.upper() if opts.get("uppercase") else text,
    }


# =============================================================================
# Part 3: Level 3 - Dataclass Specifications
# =============================================================================

@dataclass
class TextProcessingSpec:
    """Specification for text processing."""

    min_length: int = 1
    max_length: int = 10000
    allowed_chars: str | None = None
    transform: str = "none"  # none, upper, lower, title


@dataclass
class TextProcessingResult:
    """Result of text processing."""

    original: str
    processed: str
    valid: bool
    errors: list[str] = field(default_factory=list)


@op
def process_with_spec(text: str, spec: TextProcessingSpec) -> TextProcessingResult:
    """Process text according to specification.

    Level 3: Structured specs with validation.
    """
    errors = []

    # Validate length
    if len(text) < spec.min_length:
        errors.append(f"Text too short (min: {spec.min_length})")
    if len(text) > spec.max_length:
        errors.append(f"Text too long (max: {spec.max_length})")

    # Validate characters
    if spec.allowed_chars:
        invalid = set(text) - set(spec.allowed_chars)
        if invalid:
            errors.append(f"Invalid characters: {invalid}")

    # Apply transform
    transforms = {
        "none": lambda t: t,
        "upper": lambda t: t.upper(),
        "lower": lambda t: t.lower(),
        "title": lambda t: t.title(),
    }
    transform_fn = transforms.get(spec.transform, transforms["none"])
    processed = transform_fn(text)

    return TextProcessingResult(
        original=text,
        processed=processed,
        valid=len(errors) == 0,
        errors=errors,
    )


# =============================================================================
# Part 4: Level 4 - Composable Specifications
# =============================================================================

@dataclass
class ValidationRule:
    """A single validation rule."""

    name: str
    check: Callable[[Any], bool]
    message: str


class SpecBuilder:
    """Builder pattern for composable specifications."""

    def __init__(self):
        self._rules: list[ValidationRule] = []
        self._transforms: list[Callable] = []

    def add_rule(
        self,
        name: str,
        check: Callable[[Any], bool],
        message: str,
    ) -> "SpecBuilder":
        """Add a validation rule."""
        self._rules.append(ValidationRule(name, check, message))
        return self

    def add_transform(self, fn: Callable) -> "SpecBuilder":
        """Add a transform step."""
        self._transforms.append(fn)
        return self

    def min_length(self, n: int) -> "SpecBuilder":
        """Add minimum length rule."""
        return self.add_rule(
            "min_length",
            lambda x: len(x) >= n,
            f"Must be at least {n} characters",
        )

    def max_length(self, n: int) -> "SpecBuilder":
        """Add maximum length rule."""
        return self.add_rule(
            "max_length",
            lambda x: len(x) <= n,
            f"Must be at most {n} characters",
        )

    def matches(self, pattern: str) -> "SpecBuilder":
        """Add pattern matching rule."""
        import re
        return self.add_rule(
            "pattern",
            lambda x: bool(re.match(pattern, x)),
            f"Must match pattern: {pattern}",
        )

    def uppercase(self) -> "SpecBuilder":
        """Add uppercase transform."""
        return self.add_transform(lambda x: x.upper())

    def strip(self) -> "SpecBuilder":
        """Add whitespace stripping."""
        return self.add_transform(lambda x: x.strip())

    def validate(self, value: Any) -> dict:
        """Validate a value against all rules."""
        errors = []
        for rule in self._rules:
            if not rule.check(value):
                errors.append({"rule": rule.name, "message": rule.message})

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "rules_checked": len(self._rules),
        }

    def process(self, value: Any) -> dict:
        """Validate and transform a value."""
        validation = self.validate(value)

        if not validation["valid"]:
            return {**validation, "result": None}

        result = value
        for transform in self._transforms:
            result = transform(result)

        return {**validation, "result": result}


def main() -> None:
    """Demonstrate progressive specification building."""
    print("Progressive Specifications")
    print("=" * 50)
    print()

    # Level 1: No spec
    print("Level 1: No Specification (Any -> Any)")
    print("-" * 50)
    result = process_any("hello")
    print(f"String input: {result}")
    result = process_any({"key": "value"})
    print(f"Dict input: {result}")
    print()

    # Level 2: Type hints
    print("Level 2: Type Hints")
    print("-" * 50)
    result = process_typed("hello world", {"uppercase": True})
    print(f"Result: {result}")
    print()

    # Level 3: Dataclass spec
    print("Level 3: Dataclass Specifications")
    print("-" * 50)
    spec = TextProcessingSpec(min_length=5, max_length=100, transform="upper")
    result = process_with_spec("hello world", spec)
    print(f"Valid: {result.valid}")
    print(f"Processed: {result.processed}")

    # Test validation
    result = process_with_spec("hi", spec)  # Too short
    print(f"Too short - Valid: {result.valid}, Errors: {result.errors}")
    print()

    # Level 4: Composable builder
    print("Level 4: Composable Specification Builder")
    print("-" * 50)
    spec = (
        SpecBuilder()
        .min_length(5)
        .max_length(50)
        .strip()
        .uppercase()
    )

    test_cases = [
        "  hello world  ",
        "hi",
        "a" * 100,
    ]

    for test in test_cases:
        result = spec.process(test)
        status = "OK" if result["valid"] else "FAIL"
        print(f"  {test[:20]!r:22} -> {status}: {result.get('result', result.get('errors'))}")
    print()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Start with minimal specs for prototyping")
    print("2. Add type hints for basic safety")
    print("3. Use dataclasses for structured specs")
    print("4. Build composable specs for complex validation")
    print()
    print("Next: Explore examples/05_data_processing/ for data handling")


if __name__ == "__main__":
    main()
