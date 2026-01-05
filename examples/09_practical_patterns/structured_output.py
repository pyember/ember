"""Structured output extraction patterns.

This example demonstrates:
- JSON output parsing
- Schema validation
- Field extraction patterns
- Error recovery for malformed outputs
- Type coercion patterns

Run with:
    python examples/09_practical_patterns/structured_output.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from ember.api import op

T = TypeVar("T")


# =============================================================================
# Part 1: Basic JSON Extraction
# =============================================================================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text that may contain other content."""
    # Try to find JSON block in markdown code fence
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


@op
def parse_llm_json_output(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM output with fallback."""
    result = extract_json_from_text(text)
    if result is not None:
        return {"success": True, "data": result}
    return {"success": False, "error": "No valid JSON found", "raw": text}


def demonstrate_json_extraction() -> None:
    """Show JSON extraction from various formats."""
    print("Part 1: Basic JSON Extraction")
    print("-" * 50)

    test_cases = [
        (
            "Clean JSON",
            '{"name": "Alice", "age": 30}',
        ),
        (
            "JSON in code block",
            'Here is the result:\n```json\n{"status": "ok", "count": 5}\n```',
        ),
        (
            "JSON with surrounding text",
            'The analysis shows: {"score": 0.85, "label": "positive"} as output.',
        ),
        (
            "No JSON present",
            "This is just plain text without any structured data.",
        ),
    ]

    for name, text in test_cases:
        result = parse_llm_json_output(text)
        status = "OK" if result["success"] else "FAILED"
        print(f"  {name}: {status}")
        if result["success"]:
            print(f"    Extracted: {result['data']}")
        else:
            print(f"    Error: {result['error']}")
    print()


# =============================================================================
# Part 2: Schema Validation
# =============================================================================

@dataclass
class FieldSpec:
    """Specification for a required field."""

    name: str
    field_type: type
    required: bool = True
    default: Any = None


@dataclass
class Schema:
    """Simple schema for validation."""

    fields: List[FieldSpec] = field(default_factory=list)

    def add_field(
        self,
        name: str,
        field_type: type,
        required: bool = True,
        default: Any = None,
    ) -> "Schema":
        """Add a field to the schema."""
        self.fields.append(FieldSpec(name, field_type, required, default))
        return self

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against schema."""
        errors = []
        validated = {}

        for spec in self.fields:
            if spec.name in data:
                value = data[spec.name]
                # Type check (handle tuple of types)
                expected = spec.field_type
                if isinstance(expected, tuple):
                    type_ok = isinstance(value, expected)
                else:
                    type_ok = isinstance(value, expected)

                if not type_ok:
                    type_name = (
                        expected.__name__
                        if hasattr(expected, "__name__")
                        else str(expected)
                    )
                    errors.append(
                        f"Field '{spec.name}' expected {type_name}, "
                        f"got {type(value).__name__}"
                    )
                else:
                    validated[spec.name] = value
            elif spec.required:
                errors.append(f"Missing required field: '{spec.name}'")
            else:
                validated[spec.name] = spec.default

        return {
            "valid": len(errors) == 0,
            "data": validated if not errors else None,
            "errors": errors,
        }


def demonstrate_schema_validation() -> None:
    """Show schema-based validation."""
    print("Part 2: Schema Validation")
    print("-" * 50)

    # Define schema
    schema = (
        Schema()
        .add_field("name", str, required=True)
        .add_field("score", (int, float), required=True)
        .add_field("tags", list, required=False, default=[])
    )

    test_cases = [
        {"name": "Valid data", "score": 95, "tags": ["a", "b"]},
        {"name": "Missing optional", "score": 80},
        {"score": 70},  # Missing required name
        {"name": "Wrong type", "score": "not a number"},
    ]

    for data in test_cases:
        result = schema.validate(data)
        status = "VALID" if result["valid"] else "INVALID"
        print(f"  Input: {data}")
        print(f"  Result: {status}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"    - {error}")
        print()


# =============================================================================
# Part 3: Structured Output Prompts
# =============================================================================

@dataclass
class OutputFormat:
    """Define expected output format for LLM."""

    fields: Dict[str, str]  # field_name -> description
    example: Optional[Dict[str, Any]] = None

    def to_prompt_instructions(self) -> str:
        """Generate prompt instructions for output format."""
        lines = ["Respond with a JSON object containing:"]

        for field_name, description in self.fields.items():
            lines.append(f"- {field_name}: {description}")

        if self.example:
            lines.append("\nExample output:")
            lines.append(f"```json\n{json.dumps(self.example, indent=2)}\n```")

        return "\n".join(lines)


def demonstrate_output_format() -> None:
    """Show structured output format specification."""
    print("Part 3: Output Format Specification")
    print("-" * 50)

    format_spec = OutputFormat(
        fields={
            "sentiment": "one of: positive, negative, neutral",
            "confidence": "float between 0 and 1",
            "key_phrases": "list of important phrases from the text",
        },
        example={
            "sentiment": "positive",
            "confidence": 0.92,
            "key_phrases": ["excellent service", "highly recommend"],
        },
    )

    print("Generated prompt instructions:")
    print(format_spec.to_prompt_instructions())
    print()


# =============================================================================
# Part 4: Field Extraction Patterns
# =============================================================================

@dataclass
class FieldExtractor:
    """Extract specific fields from unstructured text."""

    patterns: Dict[str, str] = field(default_factory=dict)  # field -> regex pattern

    def add_pattern(self, field_name: str, pattern: str) -> "FieldExtractor":
        """Add an extraction pattern."""
        self.patterns[field_name] = pattern
        return self

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract all fields from text."""
        results = {}

        for field_name, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Use first capturing group if available, else full match
                results[field_name] = match.group(1) if match.groups() else match.group()
            else:
                results[field_name] = None

        return results


def demonstrate_field_extraction() -> None:
    """Show pattern-based field extraction."""
    print("Part 4: Field Extraction")
    print("-" * 50)

    extractor = (
        FieldExtractor()
        .add_pattern("email", r"[\w.+-]+@[\w-]+\.[\w.-]+")
        .add_pattern("phone", r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
        .add_pattern("date", r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
        .add_pattern("amount", r"\$[\d,]+(?:\.\d{2})?")
    )

    text = """
    Contact: john.doe@example.com
    Phone: (555) 123-4567
    Date: 12/25/2024
    Total: $1,234.56
    """

    print(f"Input text:{text}")
    print("Extracted fields:")
    results = extractor.extract(text)
    for field_name, value in results.items():
        print(f"  {field_name}: {value}")
    print()


# =============================================================================
# Part 5: Error Recovery
# =============================================================================

@dataclass
class OutputRecovery:
    """Recover from malformed outputs."""

    strategies: List[str] = field(default_factory=lambda: [
        "json_extract",
        "key_value_parse",
        "single_value",
    ])

    def recover(self, text: str, expected_fields: List[str]) -> Dict[str, Any]:
        """Attempt to recover structured data from text."""
        # Strategy 1: JSON extraction
        json_result = extract_json_from_text(text)
        if json_result:
            return {"strategy": "json_extract", "data": json_result}

        # Strategy 2: Key-value parsing
        kv_data = {}
        for field_name in expected_fields:
            pattern = rf"{field_name}\s*[:=]\s*['\"]?([^'\"\n,}}]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                kv_data[field_name] = match.group(1).strip()

        if kv_data:
            return {"strategy": "key_value_parse", "data": kv_data}

        # Strategy 3: Single value (if only one field expected)
        if len(expected_fields) == 1:
            return {
                "strategy": "single_value",
                "data": {expected_fields[0]: text.strip()},
            }

        return {"strategy": "failed", "data": None, "raw": text}


def demonstrate_error_recovery() -> None:
    """Show error recovery strategies."""
    print("Part 5: Error Recovery")
    print("-" * 50)

    recovery = OutputRecovery()

    test_cases = [
        ('{"status": "ok"}', ["status"]),
        ("status: complete, progress: 100%", ["status", "progress"]),
        ("The answer is yes", ["answer"]),
        ("Completely unstructured text here", ["field1", "field2"]),
    ]

    for text, fields in test_cases:
        result = recovery.recover(text, fields)
        display_text = f"'{text[:40]}...'" if len(text) > 40 else f"'{text}'"
        print(f"  Input: {display_text}")
        print(f"  Strategy: {result['strategy']}")
        print(f"  Recovered: {result.get('data')}")
        print()


# =============================================================================
# Part 6: Type Coercion
# =============================================================================

@dataclass
class TypeCoercer:
    """Coerce string values to expected types."""

    def coerce(self, value: Any, target_type: Type) -> Any:
        """Coerce value to target type."""
        if isinstance(value, target_type):
            return value

        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1", "on")
            return bool(value)

        if target_type == int:
            if isinstance(value, str):
                # Remove commas and whitespace
                cleaned = value.replace(",", "").strip()
                return int(float(cleaned))
            return int(value)

        if target_type == float:
            if isinstance(value, str):
                cleaned = value.replace(",", "").strip()
                return float(cleaned)
            return float(value)

        if target_type == list:
            if isinstance(value, str):
                # Try JSON parse first
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
                # Fall back to comma separation
                return [v.strip() for v in value.split(",")]
            return list(value)

        return value


def demonstrate_type_coercion() -> None:
    """Show type coercion."""
    print("Part 6: Type Coercion")
    print("-" * 50)

    coercer = TypeCoercer()

    test_cases = [
        ("true", bool),
        ("yes", bool),
        ("42", int),
        ("1,234", int),
        ("3.14", float),
        ('["a", "b", "c"]', list),
        ("a, b, c", list),
    ]

    for value, target in test_cases:
        result = coercer.coerce(value, target)
        print(f"  '{value}' -> {target.__name__}: {result} ({type(result).__name__})")
    print()


def main() -> None:
    """Demonstrate structured output patterns."""
    print("Structured Output Patterns")
    print("=" * 50)
    print()

    demonstrate_json_extraction()
    demonstrate_schema_validation()
    demonstrate_output_format()
    demonstrate_field_extraction()
    demonstrate_error_recovery()
    demonstrate_type_coercion()

    print("Key Takeaways")
    print("-" * 50)
    print("1. JSON extraction handles code blocks and embedded JSON")
    print("2. Schema validation ensures required fields and types")
    print("3. Output format specs guide LLM responses")
    print("4. Pattern-based extraction handles unstructured text")
    print("5. Recovery strategies handle malformed outputs gracefully")
    print("6. Type coercion normalizes string values to expected types")
    print()
    print("Next: Explore examples/10_evaluation_suite/ for testing patterns")


if __name__ == "__main__":
    main()
