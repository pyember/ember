"""Validation decorators and helpers with Ember-friendly ergonomics.

These wrappers sit on top of Pydantic's validator machinery so teams can build
validated :class:`EmberModel` classes without importing internal modules or
remembering library-specific names.
"""

from typing import Any, Callable, Literal, Optional, Type, TypeVar, Union

# Import pydantic validators internally
from pydantic import (
    field_validator as _pydantic_field_validator,
)
from pydantic import (
    model_validator as _pydantic_model_validator,
)

T = TypeVar("T")
FieldValidatorMode = Literal["before", "after", "wrap", "plain"]
ModelValidatorMode = Literal["before", "after", "wrap"]


def field_validator(
    *fields: str,
    check_fields: Optional[bool] = None,
    mode: FieldValidatorMode = "after",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a validator that runs on individual fields of an :class:`EmberModel`.

    Args:
        *fields: Field names to validate.
        check_fields: When ``True``, ensure the named fields exist on the model.
        mode: Pydantic validation phase (``"before"``, ``"after"``, or ``"wrap"``).

    Returns:
        Callable[[Callable], Callable]: Decorator that registers ``fn`` as a field
        validator.

    Examples:
        >>> from ember.api.types import EmberModel
        >>> class Profile(EmberModel):
        ...     username: str
        ...
        ...     @field_validator("username")
        ...     def normalize(cls, value: str) -> str:
        ...         if len(value.strip()) < 3:
        ...             raise ValueError("username too short")
        ...         return value.strip().lower()
    """
    return _pydantic_field_validator(*fields, mode=mode, check_fields=check_fields)  # type: ignore[call-overload,no-any-return]


def model_validator(
    *,
    mode: ModelValidatorMode = "after",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a validator that inspects the entire :class:`EmberModel` instance.

    Args:
        mode: Pydantic validation phase. Only ``"after"`` is supported currently.

    Returns:
        Callable[[Callable], Callable]: Decorator that registers ``fn`` as a
        model-level validator.

    Examples:
        >>> from ember.api.types import EmberModel
        >>> class Transaction(EmberModel):
        ...     amount: float
        ...     destination: str | None = None
        ...     kind: str
        ...
        ...     @model_validator()
        ...     def ensure_destination(self):
        ...         if self.kind == "transfer" and not self.destination:
        ...             raise ValueError("destination required")
        ...         return self
    """

    def decorator(func: Callable) -> Callable:
        # For model validators, we want to maintain the self-based signature
        # This is more Pythonic than Pydantic's approach
        if mode == "after":
            # The function already expects self, so we can use it directly
            return _pydantic_model_validator(mode=mode)(func)  # type: ignore[call-overload,no-any-return,return-value]
        else:
            # For other modes, we might need different transformations
            # For now, we only support "after" mode as it's the most common
            raise ValueError(f"Unsupported validation mode: {mode}")

    return decorator


def validator(
    *fields: str,
    always: bool = False,
    check_fields: Optional[bool] = None,
) -> Callable[[Callable], Callable]:
    """Legacy alias for :func:`field_validator`.

    Args:
        *fields: Field names to validate.
        always: Included for compatibility; ignored.
        check_fields: Forwarded to :func:`field_validator`.

    Returns:
        Callable[[Callable], Callable]: Decorator returned by
        :func:`field_validator`.
    """
    # Redirect to field_validator for compatibility
    return field_validator(*fields, check_fields=check_fields)


# Validation helpers that provide common validation patterns
class ValidationHelpers:
    """Factory methods that attach reusable validation logic to models."""

    @staticmethod
    def email_validator(field_name: str = "email") -> Callable[[Type[T]], Type[T]]:
        """Return a class decorator that enforces a simple email pattern.

        Args:
            field_name: Target field name. Defaults to ``"email"``.

        Returns:
            Callable[[Type[T]], Type[T]]: Decorator that attaches a validator to
            ``field_name``.

        Raises:
            ValueError: Raised when the supplied value lacks ``@`` or a domain.

        Examples:
            >>> from ember.api.types import EmberModel
            >>> @ValidationHelpers.email_validator()
            ... class Account(EmberModel):
            ...     email: str
            >>> Account(email="user@example.com").email
            'user@example.com'
        """

        def decorator(cls: Type[T]) -> Type[T]:
            @field_validator(field_name)
            def validate_email(cls: Type[T], value: str) -> str:
                value = value.strip().lower()
                if "@" not in value or "." not in value.split("@")[1]:
                    raise ValueError("Invalid email format")
                return value

            # Attach the validator to the class
            setattr(cls, f"_validate_{field_name}", validate_email)
            return cls

        return decorator

    @staticmethod
    def range_validator(
        field_name: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Return a class decorator enforcing inclusive numeric bounds.

        Args:
            field_name: Name of the numeric field to validate.
            min_value: Minimum allowed value (inclusive). ``None`` disables the check.
            max_value: Maximum allowed value (inclusive). ``None`` disables the check.

        Returns:
            Callable[[Type[T]], Type[T]]: Decorator attaching the validator.

        Raises:
            ValueError: Value falls outside of the specified bounds.

        Examples:
            >>> from ember.api.types import EmberModel
            >>> @ValidationHelpers.range_validator("score", min_value=0, max_value=1)
            ... class ScoredExample(EmberModel):
            ...     score: float
            >>> ScoredExample(score=0.5).score
            0.5
        """

        def decorator(cls: Type[T]) -> Type[T]:
            @field_validator(field_name)
            def validate_range(cls: Type[T], value: Union[int, float]) -> Union[int, float]:
                if min_value is not None and value < min_value:
                    raise ValueError(f"{field_name} must be at least {min_value}")
                if max_value is not None and value > max_value:
                    raise ValueError(f"{field_name} must be at most {max_value}")
                return value

            setattr(cls, f"_validate_{field_name}_range", validate_range)
            return cls

        return decorator

    @staticmethod
    def length_validator(
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Return a class decorator enforcing string length constraints.

        Args:
            field_name: Target field name.
            min_length: Minimum allowed length. ``None`` disables the check.
            max_length: Maximum allowed length. ``None`` disables the check.

        Returns:
            Callable[[Type[T]], Type[T]]: Decorator that adds validation logic.

        Raises:
            ValueError: Value length falls outside the configured bounds.

        Examples:
            >>> from ember.api.types import EmberModel
            >>> @ValidationHelpers.length_validator("name", min_length=3)
            ... class NamedThing(EmberModel):
            ...     name: str
            >>> NamedThing(name="Ada").name
            'Ada'
        """

        def decorator(cls: Type[T]) -> Type[T]:
            @field_validator(field_name)
            def validate_length(cls: Type[T], value: str) -> str:
                if min_length is not None and len(value) < min_length:
                    raise ValueError(f"{field_name} must be at least {min_length} characters")
                if max_length is not None and len(value) > max_length:
                    raise ValueError(f"{field_name} must be at most {max_length} characters")
                return value

            setattr(cls, f"_validate_{field_name}_length", validate_length)
            return cls

        return decorator


# Export the public API
__all__ = [
    "field_validator",
    "model_validator",
    "validator",  # Legacy compatibility
    "ValidationHelpers",
]
