import pytest
from ember.registry.operators.operator_base import (
    Operator,
    OperatorMetadata,
    OperatorType,
    Signature,
)
from ember.registry.operators.operator_registry import OperatorRegistry
from pydantic import BaseModel
from typing import Any, Dict


class SimpleInputs(BaseModel):
    query: str


class SimpleOperator(Operator[SimpleInputs, Dict[str, Any]]):
    metadata = OperatorMetadata(
        code="SIMPLE",
        description="Simple test operator",
        operator_type=OperatorType.RECURRENT,
        signature=Signature(required_inputs=["query"], input_model=SimpleInputs),
    )

    def forward(self, inputs: SimpleInputs) -> Dict[str, Any]:
        return {"echo": inputs.query}


@pytest.fixture
def registered_simple_operator():
    OperatorRegistry().register("SIMPLE", SimpleOperator)
    yield


def test_simple_operator_no_signature_check(registered_simple_operator):
    """
    Operator has a signature with a required input.
    Test passing correct input.
    """
    op_class = OperatorRegistry().get("SIMPLE")
    op = op_class()
    inp = op_class.build_inputs(query="Hello")
    result = op(inp)
    assert result["echo"] == "Hello"


def test_operator_missing_required_input(registered_simple_operator):
    """
    Missing 'query' field should raise error.
    """
    op_class = OperatorRegistry().get("SIMPLE")
    op = op_class()
    with pytest.raises(ValueError):
        op({})  # empty dict, no 'query'


def test_operator_structured_output():
    """
    Test operator that returns structured output:
    - Create a mock operator with a structured_output in the signature.
    - Validate output model enforcement.
    """

    class OutModel(BaseModel):
        final: str

    class StructOutOperator(Operator[SimpleInputs, OutModel]):
        metadata = OperatorMetadata(
            code="STRUCT_OUT",
            description="Test operator with structured output",
            operator_type=OperatorType.RECURRENT,
            signature=Signature(
                required_inputs=["query"],
                input_model=SimpleInputs,
                structured_output=OutModel,
            ),
        )

        def forward(self, inputs: SimpleInputs) -> OutModel:
            return {"final": inputs.query.upper()}

    OperatorRegistry().register("STRUCT_OUT", StructOutOperator)
    op_class = OperatorRegistry().get("STRUCT_OUT")
    op = op_class()
    inp = op_class.build_inputs(query="hello")
    res = op(inp)
    assert res.final == "HELLO"


def test_operator_persona_prompt_template():
    """
    Test operator with persona and prompt_template:
    - Mock operator that uses persona from config and a prompt_template.
    - Check final prompt correctness.
    TODO: Once an operator that uses persona and a template is defined, implement test.
    """
    pass


def test_operator_plan_mode():
    """
    Test operator that returns an ExecutionPlan:
    - Create a mock operator with to_plan().
    - Verify that calling op with inputs runs the plan.
    - Check combine_plan_results behember.
    TODO: Implement a mock operator with to_plan and multiple tasks.
    """
    pass
