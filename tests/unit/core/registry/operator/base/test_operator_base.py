"""
Tests for the core Operator base functionality.

This module verifies:
    - Execution of a dummy operator.
    - Registration of sub-operators.
    - Construction of inputs when an input model is defined.
    - Proper error handling when specifications are missing or misconfigured.
"""

from typing import Any, Dict, Optional, Type

import pytest
from pydantic import BaseModel

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.operator.exceptions import OperatorExecutionError
from ember.core.registry.prompt_specification.specification import Specification


class DummyInput(BaseModel):
    """Typed input model for the dummy operator.

    Attributes:
        value (int): The numerical value provided as input.
    """

    value: int


class DummyOutput(BaseModel):
    """Typed output model for the dummy operator.

    Attributes:
        result (int): The resulting value computed by the operator.
    """

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
