import pytest
from avior.graph import NoNGraphData, GraphExecutor
from avior.registry.operators.operator_base import (
    Operator,
    OperatorMetadata,
    OperatorType,
    Signature,
)
from pydantic import BaseModel
from avior.registry.operators.operator_registry import OperatorRegistry


class SimpleQueryInputs(BaseModel):
    query: str


class SimpleQueryOperator(Operator[SimpleQueryInputs, Dict[str, Any]]):
    metadata = OperatorMetadata(
        code="SIMPLEQ",
        description="Simple query operator",
        operator_type=OperatorType.RECURRENT,
        signature=Signature(required_inputs=["query"], input_model=SimpleQueryInputs),
    )

    def forward(self, inputs: SimpleQueryInputs):
        return {"answer": inputs.query + "!"}


@pytest.fixture
def registered_simpleq():
    OperatorRegistry().register("SIMPLEQ", SimpleQueryOperator)
    yield


def test_graph_executor_single_node(registered_simpleq):
    from avior.graph import GraphNode

    graph_data = NoNGraphData()
    op_class = OperatorRegistry().get("SIMPLEQ")
    op = op_class()
    graph_data.add_node("node1", op, [])
    executor = GraphExecutor()
    result = executor.execute(graph_data, {"query": "Hello"})
    assert result["answer"] == "Hello!"


def test_graph_executor_cycle(registered_simpleq):
    """
    Introduce a cycle and expect ValueError.
    """
    from avior.graph import GraphNode

    graph_data = NoNGraphData()
    op_class = OperatorRegistry().get("SIMPLEQ")
    op = op_class()
    graph_data.add_node("n1", op, ["n2"])
    graph_data.add_node("n2", op, ["n1"])
    executor = GraphExecutor()
    with pytest.raises(ValueError):
        executor.execute(graph_data, {"query": "Data"})


def test_graph_executor_multi_node(registered_simpleq):
    """
    Multiple nodes:
    1) ENSEMBLE node producing responses
    2) MOST_COMMON node depending on ENSEMBLE
    Validate final output correctness.

    TODO: Once ENSEMBLE and MOST_COMMON are registered, build graph and test final result.
    """
    pass
