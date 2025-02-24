"""
Core Operator abstraction for the Ember framework.

An Operator in Ember is an immutable, pure function with an associated Signature.
Operators are implemented as frozen dataclasses (extending EmberModule) and are
automatically registered with the PyTree system for transformation operations.

Key features of the Operator system:
1. Input and output validation through Signatures
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

from src.ember.core.registry.operator.base._module import EmberModule
from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.operator.exceptions import (
    OperatorSignatureNotDefinedError,
    OperatorExecutionError,
)

logger = logging.getLogger(__name__)

# Type variables for input and output models, bounded to BaseModel
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class Operator(EmberModule, Generic[T_in, T_out], abc.ABC):
    """Abstract base class for operators in the Ember framework.

    An Operator is an immutable, functional component that transforms inputs to outputs
    according to its *signature*.

    Operators also implement a `forward` method. 
    This design is intended to adhere to functional programming 
    style, emphasizing immutability, explicit interfaces, and composability.

    Attributes:
        signature (ClassVar[Signature]): Contains the operator's input/output signature specification.
            Subclasses must define this class variable with a valid Signature instance.
    """

    # Class variable to be overridden by subclasses
    signature: ClassVar[Signature]

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

    def __call__(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Executes the operator with automatic input and output validation.

        This method orchestrates the execution flow:
        1. Validates inputs against the signature's input model
        2. Calls the forward method with validated inputs
        3. Validates the output against the signature's output model

        Args:
            inputs (Union[T_in, Dict[str, Any]]): Either an already validated input model (T_in)
                or a raw dictionary that will be validated against the input model.

        Returns:
            T_out: The validated, computed output conforming to the output model.

        Raises:
            OperatorSignatureNotDefinedError: If this operator has no valid signature defined.
            SignatureValidationError: If input or output validation fails.
            OperatorExecutionError: If any error occurs in the forward computation.
        """
        # Retrieve and validate the signature
        try:
            signature: Signature = self.signature
        except OperatorSignatureNotDefinedError as e:
            raise OperatorSignatureNotDefinedError(
                message="Operator signature must be defined."
            ) from e

        try:
            # Validate inputs if necessary
            validated_inputs: T_in = (
                signature.validate_inputs(inputs=inputs)
                if isinstance(inputs, dict)
                else inputs
            )

            # Execute the core computation
            operator_output: T_out = self.forward(inputs=validated_inputs)

            # Validate output
            validated_output: T_out = signature.validate_output(output=operator_output)
            return validated_output
            
        except Exception as e:
            # Catch any errors during execution and wrap them
            if not isinstance(e, OperatorSignatureNotDefinedError):
                raise OperatorExecutionError(
                    message=f"Error executing operator {self.__class__.__name__}: {str(e)}"
                ) from e
            raise

    @property
    def signature(self) -> Signature:
        """Property accessor for the operator's signature.

        Returns:
            Signature: The operator's associated signature.

        Raises:
            OperatorSignatureNotDefinedError: If the signature has not been defined.
        """
        # Look up the 'signature' in the subclass's dict
        subclass_sig = type(self).__dict__.get("signature", None)
        
        # Validate that the signature is properly defined
        if subclass_sig is None or subclass_sig is Operator.__dict__.get("signature"):
            raise OperatorSignatureNotDefinedError(
                message="Operator signature must be defined."
            )
        return subclass_sig
