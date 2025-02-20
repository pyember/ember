"""Core Operator Abstraction for Ember.

An Operator is an immutable, pure function with an associated Signature.
Operators are implemented as frozen dataclasses and automatically registered
as PyTree nodes. Subclasses must implement forward() to define their computation.
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
    This aligns with our specifications model, conducive to rigor and clarity.

    Operator extends EmberModule, which automatically registers the operator as a PyTree node.
    This allows operators to be used in XCS execution plans, traced, etc.

    Attributes:
        signature: Contains the operator's input/output signature spec.
    """

    signature: Signature

    # --------------------------------------------------------------------------
    # Core Computation
    # --------------------------------------------------------------------------
    @abc.abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Performs the operator's primary computation.

        Args:
            inputs: Validated input data.

        Returns:
            The computed output.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    # --------------------------------------------------------------------------
    # Signature Access
    # --------------------------------------------------------------------------
    def get_signature(self) -> Signature:
        """Returns the operator's signature.

        Returns:
            The signature instance associated with this operator.

        Raises:
            OperatorSignatureNotDefinedError: If the signature is not defined.
        """
        if self.signature is None:
            raise OperatorSignatureNotDefinedError()
        return self.signature

    # --------------------------------------------------------------------------
    # Callable Interface
    # --------------------------------------------------------------------------
    def __call__(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Executes the operator.

        Workflow:
          1. Validate inputs using the operator's signature.
          2. Execute the operator's forward computation.
          3. Validate and return the output.

        Args:
            inputs: Raw or prevalidated input data.

        Returns:
            The validated output.

        Raises:
            OperatorSignatureNotDefinedError: If the signature is not defined.
            SignatureValidationError: If input or output validation fails.
            OperatorExecutionError: If an error occurs during forward computation.
        """
        if self.signature is None:
            raise OperatorSignatureNotDefinedError()

        try:
            validated_inputs: T_in = self.signature.validate_inputs(inputs=inputs)
        except Exception as e:
            raise SignatureValidationError("Input validation failed.") from e

        try:
            raw_output = self.forward(inputs=validated_inputs)
        except Exception as e:
            logger.exception("Error during operator forward computation.")
            raise OperatorExecutionError(str(e)) from e

        try:
            validated_output = self.signature.validate_output(output=raw_output)
        except Exception as e:
            raise SignatureValidationError("Output validation failed.") from e
        return validated_output

