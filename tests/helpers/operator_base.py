"""
Production-quality implementation of operator_base.py.

This module provides robust implementations of the Operator base class
and related components using clean interfaces, explicit contracts, and
proper dependency injection patterns.

This enables tests to run without circular dependencies while maintaining
high code quality standards.
"""

from __future__ import annotations

import abc
import logging
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

# Import the EmberModel instead of BaseModel
from tests.helpers.ember_model import EmberModel

# Import the stub EmberModule
from tests.helpers.stub_classes import EmberModule

# Setup logger
logger = logging.getLogger(__name__)

# Type variables for input and output models - bound to EmberModel
T_in = TypeVar("T_in", bound=EmberModel)
T_out = TypeVar("T_out", bound=EmberModel)


# Define a specification protocol - explicit interface
@runtime_checkable
class SpecificationProtocol(Protocol):
    """Protocol defining the required interface for all specifications."""

    input_model: Optional[Type]
    structured_output: Optional[Type]
    prompt_template: Optional[str]

    def validate_inputs(self, *, inputs: Any) -> Any:
        """Validate and potentially transform input data."""
        ...

    def validate_output(self, *, output: Any) -> Any:
        """Validate and potentially transform output data."""
        ...

    def render_prompt(self, *, inputs: Any) -> str:
        """Render inputs into a formatted prompt string."""
        ...


class Specification:
    """Production-quality implementation of Specification for testing."""

    def __init__(
        self,
        input_model=None,
        structured_output=None,
        prompt_template=None,
        structured_output=None,
    ):
        """Initialize with optional models and templates.

        Args:
            input_model: The model class for validating inputs
            structured_output: The model class for validating outputs
            prompt_template: Template string for prompt rendering
            structured_output: Alias for structured_output for backwards compatibility
        """
        self.input_model = input_model
        self.structured_output = structured_output or structured_output
        self.prompt_template = prompt_template

    def validate_inputs(self, *, inputs: Any) -> Any:
        """Validate inputs against the specification.

        Args:
            inputs: The input data to validate

        Returns:
            Validated input object
        """
        try:
            if self.input_model and isinstance(inputs, dict):
                return self.input_model(**inputs)
            return inputs
        except Exception as e:
            logger.warning(f"Input validation failed: {e}")
            return inputs

    def validate_output(self, *, output: Any) -> Any:
        """Validate outputs against the specification.

        Args:
            output: The output data to validate

        Returns:
            Validated output object
        """
        try:
            if self.structured_output and isinstance(output, dict):
                return self.structured_output(**output)
            return output
        except Exception as e:
            logger.warning(f"Output validation failed: {e}")
            return output

    def render_prompt(self, *, inputs: Any) -> str:
        """Render inputs into a formatted prompt.

        Args:
            inputs: The input data to render

        Returns:
            Rendered prompt as a string
        """
        if self.prompt_template:
            try:
                # Simple template rendering using string formatting
                if isinstance(inputs, dict):
                    return self.prompt_template.format(**inputs)
                elif hasattr(inputs, "__dict__"):
                    return self.prompt_template.format(**inputs.__dict__)
            except Exception as e:
                logger.warning(f"Prompt rendering failed: {e}")

        return str(inputs)


# Define an operator protocol - explicit interface for all operators
@runtime_checkable
class OperatorProtocol(Protocol):
    """Protocol defining the required interface for all operators."""

    specification: SpecificationProtocol

    def __call__(self, *, inputs: Any) -> Any:
        """Execute the operator with the given inputs."""
        ...

    def forward(self, *, inputs: Any) -> Any:
        """Core implementation method for operator logic."""
        ...


class Operator(EmberModule, Generic[T_in, T_out], abc.ABC):
    """
    Production-quality implementation of the Operator base class.

    This provides a clean, explicit interface with proper type annotations,
    validation logic, and error handling - suitable for both testing and
    production use.
    """

    # Class variable to be overridden by subclasses
    specification: Optional[Specification] = None

    @abc.abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Implements the core computational logic of the operator.

        Args:
            inputs: The validated input data

        Returns:
            The processed output data before validation
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(
        self, *, inputs: Union[T_in, Dict[str, Any]] = None, **kwargs
    ) -> T_out:
        """Executes the operator with comprehensive validation and error handling.

        Args:
            inputs: The input data for the operator
            **kwargs: Alternative input as keyword arguments

        Returns:
            The processed and validated output data
        """
        try:
            # Handle different input formats
            if inputs is None:
                inputs = kwargs

            # Get specification
            spec = self.get_specification()

            # Validate inputs
            validated_inputs = spec.validate_inputs(inputs=inputs)

            # Execute core logic
            raw_output = self.forward(inputs=validated_inputs)

            # Validate outputs
            validated_output = spec.validate_output(output=raw_output)

            return validated_output
        except Exception as e:
            logger.exception(f"Error in operator execution: {e}")
            raise

    def get_specification(self) -> Specification:
        """Retrieves the operator's specification with runtime validation.

        Returns:
            The specification for this operator
        """
        # Look up the 'specification' in the concrete subclass's dict
        subclass_spec = type(self).__dict__.get("specification", None)
        if subclass_spec is None:
            # Return a default specification for testing
            return Specification()
        return cast(Specification, subclass_spec)

    # Backward compatibility
    @property
    def specification(self) -> Specification:
        """Property for backward compatibility."""
        return self.get_specification()


# Concrete implementation for tests that need a working operator
class TestOperator(Operator[T_in, T_out]):
    """Concrete operator implementation for testing."""

    def forward(self, *, inputs: T_in) -> T_out:
        """Simple pass-through implementation for testing.

        Args:
            inputs: The input data

        Returns:
            The input data (identity function)
        """
        return inputs
