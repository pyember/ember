"""Handle errors gracefully in Ember applications.

This example demonstrates:
- Exception types in Ember
- Try/except patterns for API calls
- Graceful degradation strategies
- Error logging best practices

Run with:
    python examples/02_core_concepts/error_handling.py
"""

from __future__ import annotations

from ember.api import op


# =============================================================================
# Part 1: Custom Exceptions
# =============================================================================

class ProcessingError(Exception):
    """Raised when an operator cannot process its input."""

    def __init__(self, message: str, input_data: str | None = None):
        super().__init__(message)
        self.input_data = input_data


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


# =============================================================================
# Part 2: Operators with Error Handling
# =============================================================================

@op
def safe_parse_number(text: str) -> dict:
    """Parse a number from text with error handling.

    Returns a result dict with success/error status.
    This pattern is useful for batch processing where you
    don't want one error to stop everything.
    """
    text = text.strip()

    if not text:
        return {"success": False, "error": "Empty input", "value": None}

    try:
        # Try integer first
        value = int(text)
        return {"success": True, "value": value, "type": "integer"}
    except ValueError:
        pass

    try:
        # Try float
        value = float(text)
        return {"success": True, "value": value, "type": "float"}
    except ValueError:
        return {"success": False, "error": f"Cannot parse: {text!r}", "value": None}


@op
def validate_and_process(data: dict) -> dict:
    """Process data with validation.

    Demonstrates the validate-then-process pattern.
    """
    # Validation phase
    required_fields = ["name", "value"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")

    if not isinstance(data["value"], (int, float)):
        raise ValidationError(f"'value' must be numeric, got {type(data['value']).__name__}")

    # Processing phase (only reached if validation passes)
    result = {
        "name": data["name"],
        "value": data["value"],
        "doubled": data["value"] * 2,
        "status": "processed",
    }
    return result


# =============================================================================
# Part 3: Error Recovery Patterns
# =============================================================================

def process_with_fallback(items: list[str]) -> list[dict]:
    """Process items with fallback for failures.

    This pattern is useful when you want to process as much
    as possible, even if some items fail.
    """
    results = []
    errors = []

    for item in items:
        try:
            result = safe_parse_number(item)
            results.append(result)
        except Exception as e:
            errors.append({"item": item, "error": str(e)})
            results.append({"success": False, "error": str(e), "value": None})

    return results


def process_with_retry(func, *args, max_attempts: int = 3, **kwargs):
    """Retry a function on failure.

    Simple retry without backoff - for API calls, add exponential backoff.
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{max_attempts} failed: {e}")

    raise last_error  # type: ignore[misc]


# =============================================================================
# Part 4: Demonstration
# =============================================================================

def demonstrate_result_pattern() -> None:
    """Show the result dict pattern for error handling."""
    print("Result Pattern (success/error dict)")
    print("-" * 50)

    test_inputs = ["42", "3.14", "hello", "", "  100  "]
    for text in test_inputs:
        result = safe_parse_number(text)
        if result["success"]:
            print(f"  {text!r:12} -> {result['value']} ({result['type']})")
        else:
            print(f"  {text!r:12} -> Error: {result['error']}")
    print()


def demonstrate_exception_handling() -> None:
    """Show try/except patterns."""
    print("Exception Handling")
    print("-" * 50)

    test_data = [
        {"name": "valid", "value": 10},
        {"name": "invalid_type", "value": "not a number"},
        {"value": 5},  # Missing 'name'
    ]

    for data in test_data:
        try:
            result = validate_and_process(data)
            print(f"  {data} -> OK: {result['doubled']}")
        except ValidationError as e:
            print(f"  {data} -> ValidationError: {e}")
        except Exception as e:
            print(f"  {data} -> Unexpected: {type(e).__name__}: {e}")
    print()


def demonstrate_batch_with_errors() -> None:
    """Show batch processing with partial failures."""
    print("Batch Processing with Errors")
    print("-" * 50)

    items = ["1", "2", "bad", "4", "also bad", "6"]
    results = process_with_fallback(items)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"  Processed {len(items)} items: {successful} succeeded, {failed} failed")
    print(f"  Values: {[r['value'] for r in results if r['success']]}")
    print()


def main() -> None:
    """Demonstrate error handling patterns."""
    print("Ember Error Handling")
    print("=" * 50)
    print()

    demonstrate_result_pattern()
    demonstrate_exception_handling()
    demonstrate_batch_with_errors()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use result dicts for batch processing (success/error pattern)")
    print("2. Use exceptions for validation and early exit")
    print("3. Implement fallbacks for graceful degradation")
    print("4. Add retry with backoff for API calls")
    print()
    print("Next: See type_safety.py for compile-time checks")


if __name__ == "__main__":
    main()
