"""
Operator System: Core Computational Abstraction

The Operator system represents the fundamental computational unit in Ember's architecture.
It implements a principled approach to functional programming with strong typing, input/output
validation, immutability guarantees, and composition patterns.

Architectural Philosophy:
- Pure Functions: Operators are stateless, deterministic transformations from input to output
- Strong Type Safety: Generic type parameters and Pydantic validation ensure correctness
- Composition First: Designed specifically for transparent composition at any scale
- Immutability: All operators are immutable after construction for thread safety
- Explicit Interface: Clear input/output contracts enforced through specifications

These design principles enable several powerful capabilities:
1. Reliable Composition: Operators can be composed without unexpected interactions
2. Safe Parallelization: Immutability guarantees make parallel execution safe
3. Automatic Validation: Input/output validation catches errors at interface boundaries
4. Transparent Transformation: Auto-registration enables higher-order transformations
5. Predictable Behavior: Deterministic execution simplifies testing and reasoning

The system intentionally separates specification (what inputs/outputs are valid) from
implementation (how computation occurs). This separation of concerns allows for
independent evolution of validation rules and computational logic.

Core Design Patterns:
- Template Method: Abstract forward() with concrete __call__() implementation
- Strategy: Operators implement different computational strategies with common interface
- Specification: Runtime validation with detailed error information
- Generic Programming: Type variables enable rich compile-time checking

Relationship to FP/FaaS:
The Operator system draws inspiration from functional programming and serverless
architectures, treating computation as pure functions with explicit contracts.
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
    """
    Abstract base class for all computational operators in Ember.
    
    Operators are immutable, validated transformations from typed inputs to typed outputs.
    Subclasses implement the forward() method with their core logic, while this base
    class handles input/output validation and error management.
    
    The class uses a Template Method pattern: __call__() orchestrates execution flow
    while forward() provides the specific implementation.
    
    Attributes:
        specification (ClassVar[Specification]): Defines the input/output contract
            that all subclasses must provide.
    """

    # Class variable to be overridden by subclasses
    specification: ClassVar[Specification]

    @abc.abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """
        Implements the core computational logic of the operator.
        
        This abstract method represents the heart of the Template Method pattern,
        defining the customization point for concrete operator implementations.
        Subclasses must implement this method to provide their specific computational
        logic while inheriting the standardized validation and execution flow from
        the base class.
        
        The forward method is guaranteed to receive validated inputs that conform
        to the operator's input model specification, removing the need for defensive
        validation code within implementations. Similarly, the return value will be
        automatically validated against the output model specification, ensuring
        consistent interface contracts.
        
        Implementation Requirements:
        1. Must be stateless - no modification of instance variables
        2. Must be idempotent - repeated calls with same inputs yield same outputs
        3. Must not have side effects - computation only affects return value
        4. Must handle all input fields defined in specification.input_model
        5. Must return value conforming to specification.output_model
        
        Optimizations:
        - forward() can assume inputs are valid (already validated by __call__)
        - Type conversions should happen here rather than in caller
        - Complex computations can be memoized (though instances should remain immutable)
        
        Args:
            inputs: Validated input data guaranteed to conform to the operator's
                  input model specification. Never null or invalid.
                  
        Returns:
            The computation result, which will be automatically validated against
            the operator's output model specification before being returned to caller.
            
        Raises:
            NotImplementedError: Abstract method that must be implemented by subclasses.
            OperatorExecutionError: For any errors during computation (will be caught and wrapped)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(
        self, *, inputs: Union[T_in, Dict[str, Any]] = None, **kwargs
    ) -> T_out:
        """
        Executes the operator with comprehensive validation and error handling.
        
        This method implements the Template Method pattern, providing a standardized
        execution flow surrounding the subclass-specific forward() implementation.
        It manages the complete lifecycle of an operator invocation:
        
        1. Input Resolution: Determines the input format and normalizes it
        2. Input Validation: Ensures inputs conform to the specification
        3. Computation: Delegates to forward() for the core logic
        4. Output Validation: Ensures results conform to the specification
        5. Error Handling: Catches and wraps all execution errors
        
        The design supports multiple invocation patterns for maximum flexibility:
        
        A. Pre-validated model invocation:
           ```python
           model = MyInputModel(field1="value", field2=123)
           result = operator(inputs=model)
           ```
        
        B. Dictionary-based invocation with validation:
           ```python
           result = operator(inputs={"field1": "value", "field2": 123})
           ```
        
        C. Keyword argument invocation:
           ```python
           result = operator(field1="value", field2=123)
           ```
        
        This flexibility maintains type safety while accommodating different
        calling patterns, enabling both strongly-typed programming and
        convenient dynamic usage.
        
        Args:
            inputs: Either a pre-validated input model instance, or a dictionary
                  of values to be validated against the input model schema.
                  If None, kwargs will be used instead.
            **kwargs: Key-value pairs representing the input fields. Only used
                     when inputs parameter is None. Provides a more Pythonic
                     calling convention for simple cases.
        
        Returns:
            The validated computation result conforming to the output model specification.
            Type checking guarantees this will be an instance of T_out.
            
        Raises:
            OperatorSpecificationNotDefinedError: If the operator class does not define
                                                a valid specification class variable.
            SpecificationValidationError: If inputs fail validation against the input model
                                        or outputs fail validation against the output model.
            OperatorExecutionError: Wrapper for any exceptions occurring during forward execution.
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
        """
        Retrieves the operator's specification with runtime validation.
        
        This property accessor provides a safe, verified reference to the operator's
        specification, ensuring that all operations that depend on the specification
        will fail safely if it has not been properly defined. The implementation
        inspects the class hierarchy to find the concrete specification, checking
        that it has been correctly overridden by the subclass.
        
        The accessor implements defensive programming by:
        1. Looking up the specification in the concrete class's dictionary
        2. Verifying it is not None or the abstract base class definition
        3. Raising a descriptive error if the specification is missing or invalid
        
        This strict validation prevents subtle bugs that could occur if an operator
        was mistakenly defined without a proper specification.
        
        Technical Implementation Notes:
        - Uses type(self).__dict__ to inspect only the concrete class's attributes
        - Compares against Operator.__dict__ to check for proper override
        - Raises specific, descriptive exception for easy debugging
        
        Returns:
            The operator's concrete specification instance, guaranteed to be
            a valid Specification instance defined by the concrete subclass.
            
        Raises:
            OperatorSpecificationNotDefinedError: If the subclass has not properly
                defined the specification class variable, or if it mistakenly
                inherited the abstract base class definition.
        """
        # Look up the 'specification' in the concrete subclass's dict
        subclass_spec = type(self).__dict__.get("specification", None)

        # Comprehensive validation of proper specification definition
        if subclass_spec is None or subclass_spec is Operator.__dict__.get("specification"):
            raise OperatorSpecificationNotDefinedError(
                message="Operator specification must be explicitly defined in the concrete class."
            )
        return subclass_spec
        
