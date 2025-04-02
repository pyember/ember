#!/usr/bin/env python3
"""Integration test for Ember high-level API.

This script tests the integrated functionality of different high-level API modules,
especially the newly implemented evaluation module.
"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

print("Testing Ember High-Level API Integration...")

# Import high-level API components
from ember.api import (
    # Data components
    DatasetBuilder,
    datasets,
    # Evaluation components
    eval,
    # Model and operator components
    models,
    non,
    operator,
)


def test_data_eval_integration():
    """Test integration between data and evaluation APIs."""
    print("\n1. Testing Data and Evaluation Integration")

    # Create a mock dataset
    mock_data = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is the square root of 16?", "answer": "4"},
    ]

    # Define a simple prediction model
    def mock_model(question):
        answers = {
            "What is 2+2?": "4",
            "What is the capital of France?": "Paris",
            "What is the square root of 16?": "5",  # Intentionally wrong
        }
        return answers.get(question, "I don't know")

    # Create a simple dataset wrapper
    class SimpleDataset:
        def __init__(self, data):
            self.entries = data

        def __iter__(self):
            return iter(self.entries)

        def __len__(self):
            return len(self.entries)

    # Build dataset
    dataset = SimpleDataset(mock_data)

    # Create evaluators
    exact_match = eval.Evaluator.from_registry("exact_match")

    # Build an evaluation pipeline
    pipeline = eval.EvaluationPipeline([exact_match])

    # Evaluate
    print("Evaluating mock model on dataset...")
    results = pipeline.evaluate(mock_model, dataset)

    # Validate results
    print(f"Evaluation results: {results}")
    accuracy = results.get("is_correct", 0)
    assert (
        abs(accuracy - 0.6667) < 0.01
    ), f"Expected accuracy around 66.7%, got {accuracy*100:.1f}%"
    print("Data and evaluation integration test successful!")


def test_model_eval_integration():
    """Test model registry and evaluation integration."""
    # This would use real models - for testing we'll simulate with a mock
    print("\n2. Testing Model Registry/Evaluation Integration")

    # In a real implementation we would use:
    # real_model = models.from_enum(ModelEnum.GPT4)

    # For testing, we'll use a mock
    class MockModelAPI:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt):
            # Mock responses
            if "capital of France" in prompt:
                return "Paris"
            if "meaning of life" in prompt:
                return "42"
            return "I don't know"

    # Mock the model registry lookup
    models.from_name = lambda name: MockModelAPI(name)

    # Get a model
    model = models.from_name("mock_gpt4")

    # Define a custom evaluator
    def contains_evaluator(prediction, reference):
        return {"contains": reference.lower() in prediction.lower()}

    # Create the evaluator
    custom_eval = eval.Evaluator.from_function(contains_evaluator)

    # Test with simple questions
    questions = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is the meaning of life?", "answer": "42"},
    ]

    # Adapt model to match expected interface
    def model_adapter(question):
        return model.generate(question)

    # Create an evaluation pipeline
    pipeline = eval.EvaluationPipeline([custom_eval])

    # Evaluate
    results = pipeline.evaluate(model_adapter, questions)
    print(f"Model evaluation results: {results}")
    assert "contains" in results, "Missing contains metric"

    print("Model and evaluation integration test successful!")


def test_custom_evaluator_integration():
    """Test custom evaluator implementation and registration."""
    print("\n3. Testing Custom Evaluator Implementation")

    # Register available evaluators
    evaluators = eval.list_available_evaluators()
    print(f"Available evaluators before: {evaluators}")

    # Create a custom evaluator class
    class LengthEvaluator(eval.IEvaluator):
        def __init__(self, min_length=10):
            self.min_length = min_length

        def evaluate(self, system_output, correct_answer, **kwargs):
            output_len = len(str(system_output))
            is_long_enough = output_len >= self.min_length
            return eval.EvaluationResult(
                is_correct=is_long_enough,
                score=min(1.0, output_len / 50),  # Normalize
                metadata={"length": output_len},
            )

    # Register the evaluator
    eval.register_evaluator("length_check", LengthEvaluator)

    # Check it was added
    evaluators = eval.list_available_evaluators()
    print(f"Available evaluators after: {evaluators}")
    assert "length_check" in evaluators, "Custom evaluator not registered"

    # Create an evaluator instance with parameters
    length_eval = eval.Evaluator.from_registry("length_check", min_length=15)

    # Test the evaluator
    short_result = length_eval.evaluate("Short", "Any")
    long_result = length_eval.evaluate(
        "This is a longer response that exceeds the minimum", "Any"
    )

    print(f"Short text evaluation: {short_result}")
    print(f"Long text evaluation: {long_result}")

    assert (
        short_result["is_correct"] is False
    ), "Short text incorrectly marked as correct"
    assert (
        long_result["is_correct"] is True
    ), "Long text incorrectly marked as incorrect"
    assert "length" in short_result, "Length metric missing"

    print("Custom evaluator test successful!")


# Run all tests
try:
    test_data_eval_integration()
    test_model_eval_integration()
    test_custom_evaluator_integration()

    print("\n✅ All integration tests passed successfully!")
except Exception as e:
    print(f"\n❌ Integration test failed: {e}")
    import traceback

    traceback.print_exc()
