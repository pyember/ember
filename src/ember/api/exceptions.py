"""Public re-exports of Ember exception types for user code.

The module centralizes high-level exception classes so applications can depend on
stable names even as internal modules evolve. Importing from this namespace keeps
the surface small and documented.

Examples:
    >>> from ember.api import exceptions
    >>> try:
    ...     models("gpt-4", "Hello")
    ... except exceptions.ModelNotFoundError:
    ...     handle_missing_model()
"""

# Import from internal modules
from ember._internal.exceptions import (
    # Configuration exceptions
    ConfigError,
    ConfigValidationError,
    # Data exceptions
    DataError,
    DataLoadError,
    DatasetNotFoundError,
    DataValidationError,
    # Base exceptions
    EmberError,
    EmberException,  # Legacy alias
    ErrorGroup,
    InitializationError,
    InvalidArgumentError,
    InvalidPromptError,
    MissingConfigError,
    # Model exceptions
    ModelError,
    ModelNotFoundError,
    # Operator exceptions
    OperatorError,
    OperatorExecutionError,
    ProviderAPIError,
    ProviderConfigError,
    SpecificationValidationError,
    # Core exceptions
    ValidationError,
)

# Create legacy aliases for backward compatibility
OperatorException = OperatorError
ModelException = ModelError
ValidationException = ValidationError

__all__ = [
    # Base exceptions
    "EmberError",
    "EmberException",
    "ErrorGroup",
    # Core exceptions
    "ValidationError",
    "ValidationException",  # Legacy alias
    "InvalidArgumentError",
    "InitializationError",
    # Model exceptions
    "ModelError",
    "ModelException",  # Legacy alias
    "ModelNotFoundError",
    "ProviderAPIError",
    "ProviderConfigError",
    "InvalidPromptError",
    # Operator exceptions
    "OperatorError",
    "OperatorException",  # Legacy alias
    "OperatorExecutionError",
    "SpecificationValidationError",
    # Data exceptions
    "DataError",
    "DataValidationError",
    "DataLoadError",
    "DatasetNotFoundError",
    # Configuration exceptions
    "ConfigError",
    "ConfigValidationError",
    "MissingConfigError",
]
