"""Compatibility shim for known pyasn1 tuple-constraint bugs."""

import sys
import warnings
from typing import Any, Callable

TqdmWarning: type[Warning] | None
try:
    from tqdm import TqdmWarning as _TqdmWarning
except ImportError:  # pragma: no cover
    TqdmWarning = None
else:
    TqdmWarning = _TqdmWarning
    warnings.filterwarnings(
        "ignore",
        message="IProgress not found. Please update jupyter and ipywidgets.",
        category=TqdmWarning,
    )


def patch_pyasn1() -> bool:
    """Apply tuple-handling fixes to ``pyasn1``.

    Returns:
        bool: ``True`` if the patch executed without error, ``False`` otherwise.
    """
    try:
        # Import pyasn1 modules that need patching
        from pyasn1.type import constraint, univ

        # Store original SingleValueConstraint class
        _original_svc = constraint.SingleValueConstraint

        class PatchedSingleValueConstraint(_original_svc):
            """SingleValueConstraint variant that supports tuple concatenation."""

            def __add__(self, other: Any):  # type: ignore[override]
                """Handle ``constraint + tuple`` concatenation gracefully."""
                if isinstance(other, tuple):
                    # Convert self to tuple for concatenation
                    return self._values + other
                elif isinstance(other, PatchedSingleValueConstraint):
                    # Combine values from both constraints
                    return PatchedSingleValueConstraint(*(self._values + other._values))
                else:
                    # Fallback to original behavior
                    return super().__add__(other)

            def __radd__(self, other: Any):  # type: ignore[override]
                """Support ``tuple + constraint`` concatenation."""
                if isinstance(other, tuple):
                    return other + self._values
                return NotImplemented

        # Replace the original class
        constraint.SingleValueConstraint = PatchedSingleValueConstraint

        # Patch the Boolean class definition that causes the error
        if hasattr(univ, "Boolean"):
            try:
                # Create a new Boolean class with proper constraint handling
                class PatchedBoolean(univ.Integer):
                    """Boolean type that applies the patched constraint."""

                    tagSet = univ.Boolean.tagSet
                    namedValues = univ.Boolean.namedValues

                    # Fix the constraint definition
                    subtypeSpec = univ.Integer.subtypeSpec + PatchedSingleValueConstraint(0, 1)

                    # Copy other attributes
                    encoding = "us-ascii"

                # Replace the original Boolean class
                univ.Boolean = PatchedBoolean
            except Exception:
                # If patching Boolean fails, continue anyway
                pass

        return True

    except Exception as e:
        # Log but don't fail - the patch is optional
        import logging

        logging.debug(f"pyasn1 patch failed (non-critical): {e}")
        return False


def ensure_pyasn1_compatibility() -> None:
    """Install an import hook so ``pyasn1`` receives the patch on first use."""
    # Check if pyasn1 is already imported
    if "pyasn1" in sys.modules:
        # If already imported, try to patch it anyway
        patch_pyasn1()
    else:
        # If not imported yet, set up an import hook
        import builtins

        _original_import: Callable[..., Any] = builtins.__import__

        def _patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            """Apply the patch the first time ``pyasn1.type.univ`` is imported."""
            module = _original_import(name, *args, **kwargs)

            # When pyasn1.type.univ is imported, apply our patches
            if name == "pyasn1.type.univ" or name.startswith("pyasn1.type.univ"):
                patch_pyasn1()
                # Restore original import to avoid overhead
                builtins.__import__ = _original_import

            return module

        # Install our import hook
        builtins.__import__ = _patched_import


# Auto-apply the patch when this module is imported
ensure_pyasn1_compatibility()
