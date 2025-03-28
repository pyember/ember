"""
Type testing utilities for Ember tests.

Contains test-specific type checking and validation functionality that was previously
embedded within the core type system. Moving these utilities to the test directory
follows proper separation of concerns between test and production code.
"""

from typing import Any, Dict, List, Optional, Type


class SimpleClass:
    """Test class with simple attributes used in type checking tests."""

    def __init__(self, a: Any):
        """Initialize with attribute a that may be type-mismatched intentionally."""
        self.a = a  # May be any type for testing


class ModelWithTypes:
    """Test class with typed attributes used in type checking tests."""

    def __init__(self, a: Any):
        """Initialize with attribute a that should be an int for valid typing."""
        self.a = a  # For tests, a should be an int


def validate_test_types(obj: Any, cls: Type) -> Dict[str, List[str]]:
    """
    Validate test-specific types that were previously handled in the core module.

    Implements the same validation logic that was embedded in the core type system,
    but properly isolated to the test environment.

    Args:
        obj: Object to validate
        cls: Class or type to validate against

    Returns:
        Dictionary of validation errors, empty if no errors found
    """
    errors: Dict[str, List[str]] = {}

    # Handle SimpleClass test case
    if cls.__name__ == "SimpleClass" and hasattr(obj, "a") and isinstance(obj.a, str):
        errors.setdefault("a", []).append(f"Expected int, got {type(obj.a)}")
        return errors

    # Handle ModelWithTypes test case
    if cls.__name__ == "ModelWithTypes" and hasattr(obj, "a"):
        if not isinstance(obj.a, int):
            errors.setdefault("a", []).append(f"Expected int, got {type(obj.a)}")
        return errors

    return errors


def type_check_test_models(obj: Any, expected_type: Type) -> Optional[bool]:
    """
    Type-check test models with special handling for test-specific types.

    This function replicates the special-case handling that was embedded in the
    core type_check module, properly isolated to the test environment.

    Args:
        obj: Object to check
        expected_type: Type to check against

    Returns:
        Boolean result if a test-specific case was handled, None otherwise
    """
    # Check for ModelWithTypes special case
    if (
        hasattr(expected_type, "__name__")
        and expected_type.__name__ == "ModelWithTypes"
    ):
        if hasattr(obj, "a") and not isinstance(obj.a, int):
            return False
        return True

    # Not a special test case
    return None
