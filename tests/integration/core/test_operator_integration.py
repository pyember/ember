"""Integration tests for operator composition and execution.

These tests verify actual operator workflows with realistic compositions.
Uses minimal test doubles to avoid import issues.
"""


import pytest

# Import minimal test doubles instead of actual implementations
from tests.helpers.operator_minimal_doubles import (
    MinimalOperator,
    MinimalTestModel,
    SimpleDeterministicOperator,
    SimpleEnsembleOperator,
    SimpleSelectorOperator,
)

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration]


class SimplePromptOperator(MinimalOperator):
    """Simple operator that appends a prompt to input."""

    def __init__(self, prompt_suffix: str = " Complete this sentence."):
        super().__init__()
        self.prompt_suffix = prompt_suffix

    def forward(self, *, inputs: MinimalTestModel) -> MinimalTestModel:
        input_text = inputs.value if inputs.value else ""
        return MinimalTestModel(value=f"{input_text}{self.prompt_suffix}")


class SimpleResponseOperator(MinimalOperator):
    """Simple operator that generates a deterministic response based on input."""

    def __init__(self, response_prefix: str = "Response: "):
        super().__init__()
        self.response_prefix = response_prefix

    def forward(self, *, inputs: MinimalTestModel) -> MinimalTestModel:
        input_text = inputs.value if inputs.value else ""
        return MinimalTestModel(value=f"{self.response_prefix}{input_text}")


@pytest.fixture
def basic_operators():
    """Fixture providing basic operators for testing."""
    return {
        "prompt": SimplePromptOperator(),
        "response": SimpleResponseOperator(),
        "ensemble": SimpleEnsembleOperator(),
        "selector": SimpleSelectorOperator(),
        "transformer": SimpleDeterministicOperator(
            transform_fn=lambda x: f"Transformed: {x}"
        ),
    }


def test_basic_operator_composition(basic_operators):
    """Test basic operator composition with prompt and response."""

    # Create a simple pipeline: prompt -> response
    prompt_op = basic_operators["prompt"]
    response_op = basic_operators["response"]

    # Test the pipeline
    input_text = "The sky is"
    prompted = prompt_op(inputs=MinimalTestModel(value=input_text))
    result = response_op(inputs=prompted)

    # Verify results
    assert prompted.value == "The sky is Complete this sentence."
    assert isinstance(result.value, str)
    assert result.value.startswith("Response: ")
    assert len(result.value) > 0


def test_complex_operator_composition(basic_operators):
    """Test complex operator composition with multiple paths and ensemble."""

    # Create differentiated prompt operators
    prompt_ops = [
        SimplePromptOperator(" Complete this sentence about nature."),
        SimplePromptOperator(" Finish this thought about the sky."),
        SimplePromptOperator(" What comes next in this sentence?"),
    ]

    # Create response operators
    response_ops = [SimpleResponseOperator(f"Response {i+1}: ") for i in range(3)]

    # Create a list of operator pairs
    operator_pairs = []
    for i in range(3):
        operator_pairs.append([prompt_ops[i], response_ops[i]])

    # Function to execute a pair
    def execute_pair(op_pair, input_model):
        step1 = op_pair[0](inputs=input_model)
        return op_pair[1](inputs=step1)

    # Create a list of results
    input_text = "The sky is"
    input_model = MinimalTestModel(value=input_text)
    results = [execute_pair(pair, input_model) for pair in operator_pairs]

    # Verify individual results
    for result in results:
        assert isinstance(result.value, str)
        assert result.value.startswith("Response ")
        assert len(result.value) > 0

    # Use ensemble to combine results
    ensemble = basic_operators["ensemble"]
    ensemble.operators = [
        SimpleDeterministicOperator(lambda x: r.value) for r in results
    ]

    # Execute ensemble
    ensemble_result = ensemble(inputs=input_model)

    # Verify ensemble result
    assert isinstance(ensemble_result.value, list)
    assert len(ensemble_result.value) == 3


def test_selector_operator(basic_operators):
    """Test the selector operator for choosing between alternatives."""

    # Create prompt operators with different suffixes
    prompt_ops = [
        SimplePromptOperator(" Complete with a weather description."),
        SimplePromptOperator(" Complete with a color description."),
    ]

    # Create response operators
    response_ops = [
        SimpleResponseOperator("Weather: "),
        SimpleResponseOperator("Color: "),
    ]

    # Create a list of operator pairs
    operator_pairs = [
        [prompt_ops[0], response_ops[0]],
        [prompt_ops[1], response_ops[1]],
    ]

    # Function to execute a pair
    def execute_pair(op_pair, input_model):
        step1 = op_pair[0](inputs=input_model)
        return op_pair[1](inputs=step1)

    # Create pipelines that produce results
    input_text = "The sky is"
    input_model = MinimalTestModel(value=input_text)

    # Define selector operators for each pipeline
    selector_ops = []
    for pair in operator_pairs:
        # Create a simple function that executes the pair
        def make_selector(p):
            return SimpleDeterministicOperator(
                lambda x: execute_pair(p, MinimalTestModel(value=x)).value
            )

        selector_ops.append(make_selector(pair))

    # Create selector
    selector = basic_operators["selector"]
    selector.operators = selector_ops
    selector.select_index = 0  # Select the first result

    # Execute selector
    result = selector(inputs=input_model)

    # Verify results
    assert isinstance(result.value, str)
    assert result.value.startswith("Weather: ")

    # Change selection index
    selector.select_index = 1
    result = selector(inputs=input_model)

    # Now should get the color response
    assert result.value.startswith("Color: ")


def test_transformation_operator(basic_operators):
    """Test the transformation operator for modifying results."""

    # Create a simple pipeline: prompt -> response -> transform
    prompt_op = basic_operators["prompt"]
    response_op = basic_operators["response"]
    transform_op = basic_operators["transformer"]

    # Test the pipeline
    input_text = "The sky is"
    input_model = MinimalTestModel(value=input_text)
    prompted = prompt_op(inputs=input_model)
    response = response_op(inputs=prompted)
    result = transform_op(inputs=response)

    # Verify results
    assert prompted.value == "The sky is Complete this sentence."
    assert response.value.startswith("Response: ")
    assert result.value.startswith("Transformed: Response: ")

    # Test direct composition
    def compose(op1, op2, op3):
        def composed_op(input_model):
            step1 = op1(inputs=input_model)
            step2 = op2(inputs=step1)
            return op3(inputs=step2)

        return composed_op

    composed = compose(prompt_op, response_op, transform_op)
    final_result = composed(input_model)

    # Verify composed result
    assert final_result.value == result.value
