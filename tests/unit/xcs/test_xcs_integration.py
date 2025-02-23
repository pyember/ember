"""Integration tests for XCS components.

This module tests an end-to-end workflow that covers tracing, graph compilation,
and execution using a mock operator.
"""

from typing import Any, Dict, Optional

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.xcs.engine.xcs_engine import TopologicalSchedulerWithParallelDispatch, compile_graph
from src.ember.xcs.graph.xcs_graph import XCSGraph
from src.ember.xcs.tracer.xcs_tracing import TracerContext
from src.ember.xcs.tracer.tracer_decorator import _convert_ir_graph_to_xcs_graph


class MockOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Mock operator for integration tests that doubles the input value.

    Attributes:
        signature: A dummy signature providing input model definitions and basic
                   validation methods.
    """
    # For testing, we use a simplified signature.
    signature = type(
        "DummySignature",
        (),
        {
            "input_model": dict,
            "validate_inputs": lambda self, *, inputs: inputs,
            "validate_output": lambda self, *, output: output,
            "render_prompt": lambda self, *, inputs: "dummy prompt",
        },
    )()

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the operator, doubling the input 'value'.

        Args:
            inputs (Dict[str, Any]): A dictionary containing the key 'value'
                with a numeric value.

        Returns:
            Dict[str, Any]: A dictionary with key 'result' containing the doubled value.
        """
        return {"result": inputs["value"] * 2}


def test_tracer_to_execution() -> None:
    """Tests the full execution workflow from tracing to scheduling.

    The test performs the following steps:
      1. Instantiates a MockOperator and defines a sample input.
      2. Uses the TracerContext to run a trace and generate an intermediate representation
         (IR) graph.
      3. Constructs an XCSGraph from the traced IR graph by adding nodes and edges based on
         the IR graph data.
      4. Compiles the XCSGraph into an execution plan.
      5. Executes the plan using the TopologicalSchedulerWithParallelDispatch scheduler.
      6. Asserts that the final output contains the expected doubled value.

    Returns:
        None
    """
    mock_operator: MockOperator = MockOperator()
    sample_input: Dict[str, Any] = {"value": 5}

    with TracerContext(top_operator=mock_operator, sample_input=sample_input) as tracer:
        ir_graph = tracer.run_trace()

    # Use our conversion helper to make a proper XCSGraph from the IR graph
    graph: XCSGraph = _convert_ir_graph_to_xcs_graph(
        ir_graph=ir_graph,
        operator=mock_operator,  # pass the same operator used during tracing
    )

    plan = compile_graph(graph=graph)
    scheduler = TopologicalSchedulerWithParallelDispatch()
    results: Dict[str, Any] = scheduler.run_plan(
        plan=plan, global_input=sample_input, graph=graph
    )

    # Retrieve the final output containing the expected doubled value.
    final_output: Optional[Dict[str, Any]] = next(
        (
            output
            for output in results.values()
            if isinstance(output, dict) and "result" in output
        ),
        None,
    )

    assert final_output is not None, "Expected a result in the final output."
    assert final_output == {"result": 10}, (
        f"Final output {final_output} does not match the expected "
        f"{{'result': 10}}."
    )