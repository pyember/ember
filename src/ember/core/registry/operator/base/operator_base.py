"""
Core Operator abstraction for the Ember framework.

An Operator in Ember is an immutable, pure function with an associated Specification.
Operators are implemented as frozen dataclasses (extending EmberModule) and are
automatically registered with the PyTree system for transformation operations.

Key features of the Operator system:
1. Input and output validation through Specifications
2. Immutability for thread safety and functional programming patterns
3. Automatic registration with transformation systems
4. Strong typing with generic input/output model specifications

This module provides the foundation for building composable, type-safe computational
units within the Ember framework.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Generic, TypeVar, Union, cast, Type, Optional, ClassVar

from pydantic import BaseModel

from ember.core.registry.operator.base._module import EmberModule
from ember.core.registry.prompt_specification.specification import Specification
from ember.core.registry.operator.exceptions import (
    OperatorSpecificationNotDefinedError,
    OperatorExecutionError,
)

logger = logging.getLogger(__name__)

# Type variables for input and output models, bounded to BaseModel
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class Operator(EmberModule, Generic[T_in, T_out], abc.ABC):
    """Abstract base class for operators in the Ember framework.

    An Operator is an immutable, functional component that transforms inputs to outputs
    according to its *specification*.

    Operators also implement a `forward` method.
    This design is intended to adhere to functional programming
    style, emphasizing immutability, explicit interfaces, and composability.

    Attributes:
        specification (ClassVar[Specification]): Contains the operator's input/output specification.
            Subclasses must define this class variable with a valid Specification instance.
    """

    # Class variable to be overridden by subclasses
    specification: ClassVar[Specification]

    @abc.abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Performs the operator's primary computation.

        This abstract method must be implemented by all subclasses to define the
        operator's core functionality.

        Args:
            inputs (T_in): Validated input data conforming to the operator's input model.

        Returns:
            T_out: The computed output conforming to the operator's output model.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(
        self, *, inputs: Union[T_in, Dict[str, Any]] = None, **kwargs
    ) -> T_out:
        """Executes the operator with automatic input and output validation.

        This method orchestrates the execution flow:
        1. Validates inputs against the specification's input model
        2. Calls the forward method with validated inputs
        3. Validates the output against the specification's output model and ensures it's the proper type

        The operator can be called in three ways:
        1. With a model instance: op(inputs=my_model_instance)
        2. With a dictionary: op(inputs={"key": "value"})
        3. With keyword arguments: op(key="value", another="value")

        Args:
            inputs (Union[T_in, Dict[str, Any]], optional): Either an already validated input model (T_in)
                or a raw dictionary that will be validated against the input model.
            **kwargs: Alternative to using 'inputs'. Key-value pairs that will be used to construct
                     the input model. Only used if 'inputs' is None.

        Returns:
            T_out: The validated, computed output conforming to the output model.

        Raises:
            OperatorSpecificationNotDefinedError: If this operator has no valid specification defined.
            SpecificationValidationError: If input or output validation fails.
            OperatorExecutionError: If any error occurs in the forward computation.
        """
        # Retrieve and validate the specification
        try:
            specification: Specification = self.specification
        except OperatorSpecificationNotDefinedError as e:
            raise OperatorSpecificationNotDefinedError(
                message="Operator specification must be defined."
            ) from e

        try:
            # Determine input format (model, dict, or kwargs)
            if inputs is not None:
                # Traditional 'inputs' parameter provided
                validated_inputs: T_in = (
                    specification.validate_inputs(inputs=inputs)
                    if isinstance(inputs, dict)
                    else inputs
                )
            elif kwargs and specification.input_model:
                # Using kwargs directly as input fields
                validated_inputs = specification.input_model(**kwargs)
            else:
                # Empty inputs or no input model defined
                validated_inputs = kwargs if kwargs else {}

            # Execute the core computation
            operator_output: T_out = self.forward(inputs=validated_inputs)

            # Ensure we have a proper model instance for the output
            # If we got a dict, convert it to the appropriate model
            if (
                isinstance(operator_output, dict)
                and hasattr(specification, "output_model")
                and specification.output_model
            ):
                operator_output = specification.output_model.model_validate(operator_output)

            # Final validation to ensure type consistency
            validated_output: T_out = specification.validate_output(output=operator_output)
            return validated_output

        except Exception as e:
            # Catch any errors during execution and wrap them
            if not isinstance(e, OperatorSpecificationNotDefinedError):
                raise OperatorExecutionError(
                    message=f"Error executing operator {self.__class__.__name__}: {str(e)}"
                ) from e
            raise

    @property
    def specification(self) -> Specification:
        """Property accessor for the operator's specification.

        Returns:
            Specification: The operator's associated specification.

        Raises:
            OperatorSpecificationNotDefinedError: If the specification has not been defined.
        """
        # Look up the 'specification' in the subclass's dict
        subclass_spec = type(self).__dict__.get("specification", None)

        # Validate that the specification is properly defined
        if subclass_spec is None or subclass_spec is Operator.__dict__.get("specification"):
            raise OperatorSpecificationNotDefinedError(
                message="Operator specification must be defined."
            )
        return subclass_spec
        
