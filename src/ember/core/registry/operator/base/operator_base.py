"""Core Operator Abstraction for Ember.

An Operator is an immutable, pure function with an associated Signature.
Operators are implemented as frozen dataclasses and automatically registered as PyTree nodes.
Subclasses must implement forward() to define their computation.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Generic, TypeVar, Union

from pydantic import BaseModel

from src.ember.core.registry.operator.base._module import EmberModule
from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.operator.exceptions import (
    OperatorSignatureNotDefinedError,
    SignatureValidationError,
    OperatorExecutionError,
)

logger = logging.getLogger(__name__)

# Type variables for the input and output Pydantic models.
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class Operator(EmberModule, Generic[T_in, T_out], abc.ABC):
    """Abstract base class for an operator in Ember.

    Each operator must explicitly define its Signature (with required input and output models).
    This design encourages immutability and explicitness, aligning with functional programming
    principles and drawing inspiration from frameworks such as JAX Equinox.

    Attributes:
        signature (Signature): Contains the operator's input/output signature specification.
    """

    signature: Signature

    @abc.abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Performs the operator's primary computation.

        Args:
            inputs (T_in): Validated input data.

        Returns:
            T_out: The computed output.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_signature(self) -> Signature:
        """Retrieves the operator's signature.

        Returns:
            Signature: The operator's associated signature.

        Raises:
            OperatorSignatureNotDefinedError: If the signature has not been defined.
        """
        if not getattr(self, "signature", None):
            raise OperatorSignatureNotDefinedError(
                "Operator signature must be defined."
            )
        return self.signature

    def __call__(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Executes the operator using its defined signature and forward computation.

        The execution workflow entails:
          1. Validating the provided inputs via the operator's signature.
          2. Running the forward computation with the validated inputs.
          3. Validating and returning the computed output.

        Args:
            inputs (Union[T_in, Dict[str, Any]]): The raw or prevalidated input data.

        Returns:
            T_out: The validated output after operator execution.

        Raises:
            OperatorSignatureNotDefinedError: If the operator's signature is not defined.
            SignatureValidationError: If input or output validation fails.
            OperatorExecutionError: If an error occurs during the forward computation.
        """
        signature: Signature = self.get_signature()

        # Validate inputs with detailed logging.
        try:
            validated_inputs: T_in = signature.validate_inputs(inputs=inputs)
        except Exception as error:
            operator_name: str = self.__class__.__name__
            error_message: str = (
                f"[{operator_name}] Input validation failed for inputs: {inputs}"
            )
            logger.error(error_message, extra={"operator": operator_name}, exc_info=True)
            raise SignatureValidationError(error_message) from error

        # Execute forward computation.
        try:
            operator_output: T_out = self.forward(inputs=validated_inputs)
        except Exception as error:
            contextual_message: str = (
                f"Error during operator forward computation "
                f"with validated inputs: {validated_inputs}"
            )
            # Combine your contextual info with the original error text so that
            # tests can match on strings like "Simulated LM failure."
            full_message: str = f"{contextual_message} | Original error: {error}"
            logger.exception(full_message)
            raise OperatorExecutionError(full_message) from error

        # Validate output with detailed logging.
        try:
            validated_output: T_out = signature.validate_output(output=operator_output)
        except Exception as error:
            operator_name: str = self.__class__.__name__
            error_message: str = (
                f"[{operator_name}] Output validation failed for operator output: {operator_output}"
            )
            logger.error(error_message, extra={"operator": operator_name}, exc_info=True)
            raise SignatureValidationError(error_message) from error

        return validated_output
