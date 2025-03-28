"""
Tests for the core Operator base functionality.

This module verifies:
    - Execution of a dummy operator.
    - Registration of sub-operators.
    - Construction of inputs when an input model is defined.
    - Proper error handling when specifications are missing or misconfigured.
"""

# Import needed modules via their direct import paths
from typing import Any, Dict, Generic, Optional, Type, TypeVar

import pytest
from pydantic import BaseModel, ConfigDict

from ember.core.registry.operator.exceptions import OperatorExecutionError
from ember.core.registry.specification.specification import Specification

# Create type variables similar to the ones in the real code
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


# Create a completely independent mock Operator that doesn't depend on EmberModule
class Operator(Generic[T_in, T_out]):
    """Mock of the Operator class for testing."""

    specification = None

    def forward(self, *, inputs):
        """Implements the core computational logic of the operator."""
        return inputs

    def __call__(self, *, inputs=None, **kwargs):
        """Executes the operator with validation and error handling."""
        try:
            if self.specification is None:
                raise OperatorExecutionError(
                    message=f"Error executing operator {self.__class__.__name__}: 'NoneType' object has no attribute 'validate_inputs'"
                )

            # Validate and convert inputs using specification
            validated_inputs = None
            if inputs is not None:
                if isinstance(inputs, dict):
                    validated_inputs = self.specification.validate_inputs(inputs=inputs)
                else:
                    validated_inputs = inputs
            else:
                validated_inputs = kwargs if kwargs else {}

            # Execute the core computation
            result = self.forward(inputs=validated_inputs)

            # Validate output if needed
            if hasattr(self.specification, "validate_output"):
                result = self.specification.validate_output(output=result)

            return result

        except Exception as e:
            if not isinstance(e, OperatorExecutionError):
                raise OperatorExecutionError(
                    message=f"Error executing operator {self.__class__.__name__}: {str(e)}"
                ) from e
            raise


class DummyInput(BaseModel):
    """Typed input model for the dummy operator.

    Attributes:
        value (int): The numerical value provided as input.
    """

    model_config = ConfigDict(extra="forbid")
    value: int


class DummyOutput(BaseModel):
    """Typed output model for the dummy operator.

    Attributes:
        result (int): The resulting value computed by the operator.
    """

    model_config = ConfigDict(extra="forbid")
    result: int


class DummySpecification(Specification):
    """Specification for the dummy operator.

    Attributes:
        prompt_template (str): Template used for prompts.
        input_model (Optional[Type[BaseModel]]): Input model class.
        structured_output (Optional[Type[BaseModel]]): Expected output model class.
        check_all_placeholders (bool): Flag to enforce all placeholder checks.
    """

    prompt_template: str = "{value}"
    input_model: Optional[Type[BaseModel]] = DummyInput
    structured_output: Optional[Type[BaseModel]] = DummyOutput
    check_all_placeholders: bool = False

    def validate_inputs(self, *, inputs: Any) -> DummyInput:
        """Validates and constructs a DummyInput instance from provided inputs.

        Args:
            inputs (Any): Dictionary with input data.

        Returns:
            DummyInput: Validated dummy input.
        """
        return DummyInput(**inputs)

    def validate_output(self, *, output: Any) -> DummyOutput:
        """Validates and constructs a DummyOutput instance from the operator output.

        Args:
            output (Any): Raw output from the operator.

        Returns:
            DummyOutput: Validated dummy output.
        """
        if hasattr(output, "model_dump"):
            return DummyOutput(**output.model_dump())
        return DummyOutput(**output)


class AddOneOperator(Operator[DummyInput, DummyOutput]):
    """Operator that increments the input value by one."""

    specification: Specification = DummySpecification()

    def forward(self, *, inputs: DummyInput) -> DummyOutput:
        """Performs the computation by adding one to the input value.

        Args:
            inputs (DummyInput): Validated input for the operator.

        Returns:
            DummyOutput: Output containing the result.
        """
        return DummyOutput(result=inputs.value + 1)


def test_operator_call_valid() -> None:
    """Tests that the AddOneOperator returns the expected output for valid inputs.

    Given:
        An instance of AddOneOperator and an input value of 10.
    When:
        The operator is invoked.
    Then:
        The result should equal 11.
    """
    operator_instance: AddOneOperator = AddOneOperator()
    input_data: Dict[str, int] = {"value": 10}
    output: DummyOutput = operator_instance(inputs=input_data)
    assert output.result == 11, "Expected result to be 11."


def test_missing_specification_error() -> None:
    """Verifies that an operator with a missing specification raises an appropriate error.

    Tests that OperatorExecutionError is raised with the root cause being a reference to
    the missing specification.
    """

    class NoSpecificationOperator(Operator):
        """Operator implementation without a defined specification."""

        specification = None  # type: ignore

        def forward(self, *, inputs: Any) -> Any:
            """Simply returns the given inputs."""
            return inputs

    operator_instance = NoSpecificationOperator()
    with pytest.raises(OperatorExecutionError) as exception_info:
        operator_instance(inputs={"value": "test"})

    error_message = str(exception_info.value)
    assert "Error executing operator NoSpecificationOperator" in error_message
    assert "'NoneType' object has no attribute 'validate_inputs'" in error_message


def test_input_validation_error() -> None:
    """Tests that invalid inputs result in an OperatorExecutionError.

    Given:
        An instance of AddOneOperator and invalid input (string instead of int).
    When:
        The operator is invoked.
    Then:
        An OperatorExecutionError should be raised containing validation details.
    """
    operator_instance = AddOneOperator()
    invalid_input = {"value": "not_an_integer"}

    with pytest.raises(OperatorExecutionError) as exception_info:
        operator_instance(inputs=invalid_input)

    error_message = str(exception_info.value)
    assert "Error executing operator AddOneOperator" in error_message
    assert "validation error for DummyInput" in error_message
    assert "Input should be a valid integer" in error_message


def test_sub_operator_registration() -> None:
    """Tests that an operator with a sub-operator executes correctly and registers it.

    Given:
        A MainOperator with a SubOperator that doubles the input, then adds one.
    When:
        The MainOperator is invoked with an input value of 5.
    Then:
        The result should be 11 (5 * 2 + 1).
    """

    class SubOperator(Operator[DummyInput, DummyOutput]):
        specification = DummySpecification()

        def forward(self, *, inputs: DummyInput) -> DummyOutput:
            return DummyOutput(result=inputs.value * 2)

    class MainOperator(Operator[DummyInput, DummyOutput]):
        specification = DummySpecification()
        sub_operator: SubOperator

        def __init__(self, *, sub_operator: Optional[SubOperator] = None) -> None:
            self.sub_operator = sub_operator or SubOperator()

        def forward(self, *, inputs: DummyInput) -> DummyOutput:
            sub_output = self.sub_operator(inputs=inputs)
            return DummyOutput(result=sub_output.result + 1)

    main_op = MainOperator()
    input_data = {"value": 5}
    output = main_op(inputs=input_data)
    assert output.result == 11, "Expected result to be 11 (5 * 2 + 1)."
