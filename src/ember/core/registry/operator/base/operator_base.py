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
)

logger = logging.getLogger(__name__)

T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class Operator(EmberModule, Generic[T_in, T_out], abc.ABC):
    """Abstract base class for an operator in Ember.

    Each operator must explicitly define its Signature (with required input and output models).
    This design encourages immutability and 'simple over easy' explicitness, aligning with functional programming
    principles and drawing inspiration from frameworks such as JAX.

    Attributes:
        signature (Signature): Contains the operator's input/output signature specification.
    """

    # Subclasses are expected to override this class variable.
    signature: Signature

    @abc.abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Performs the operator's primary computation.

        Args:
            inputs (T_in): Validated input data.

        Returns:
            T_out: The computed output.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Executes the operator by validating inputs, running the forward computation,
        and validating the output.

        Args:
            inputs (Union[T_in, Dict[str, Any]]): Either an already validated input model (T_in)
                or a raw dictionary that needs validation.

        Returns:
            T_out: The validated, computed output.

        Raises:
            OperatorSignatureNotDefinedError: If this operator has no valid signature defined.
            SignatureValidationError: If input or output validation fails.
            OperatorExecutionError: If any error occurs in the forward computation.
        """
        signature: Signature = getattr(self.__class__, "signature", None)
        if signature is None or not hasattr(signature, "validate_inputs"):
            raise OperatorSignatureNotDefinedError(
                "Operator signature must be defined."
            )

        # Validate inputs.
        validated_inputs: T_in = (
            signature.validate_inputs(inputs=inputs)
            if isinstance(inputs, dict)
            else inputs
        )

        # Run the forward computation.
        operator_output: T_out = self.forward(inputs=validated_inputs)

        # Validate output.
        validated_output: T_out = signature.validate_output(output=operator_output)
        return validated_output

    # --------------------------------------------------------------------------
    # Property Accessors
    # --------------------------------------------------------------------------

    @property
    def signature(self) -> Signature:
        """Property accessor for the operator's signature.

        Returns:
            Signature: The operator's associated signature.

        Raises:
            OperatorSignatureNotDefinedError: If the signature has not been defined.
        """
        # Look up the 'signature' in the subclass's dict.
        subclass_sig = type(self).__dict__.get("signature", None)
        # If the subclass did not override the base 'signature' property, it's not defined.
        if subclass_sig is None or subclass_sig is Operator.__dict__.get("signature"):
            raise OperatorSignatureNotDefinedError(
                "Operator signature must be defined."
            )
        return subclass_sig
