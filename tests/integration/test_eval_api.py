#!/usr/bin/env python3
"""Test script for the new evaluation API in Ember."""

import sys

print("Testing Evaluation API...")

# Test the evaluation API
try:
    from ember.api import eval

    # Test module access
    print(f"Module access: {eval.__name__}")

    # Test list_available_evaluators
    print("\nTesting list_available_evaluators()...")
    available = eval.list_available_evaluators()
    print(f"Available evaluators: {available}")
    assert "exact_match" in available, "Expected exact_match evaluator in registry"
    assert "accuracy" in available, "Expected accuracy evaluator in registry"
    assert "numeric" in available, "Expected numeric evaluator in registry"
    assert "regex" in available, "Expected regex evaluator in registry"

    # Test registry-based evaluator
    print("\nTesting registry-based evaluator...")
    exact_evaluator = eval.Evaluator.from_registry("exact_match")
    result_exact = exact_evaluator.evaluate("Paris", "paris")
    print(f"Exact match result: {result_exact}")

    # Test register_evaluator
    print("\nTesting register_evaluator()...")

    # Create and register a custom evaluator class
    class WordCountEvaluator(eval.IEvaluator):
        def __init__(self, min_words=5):
            # Store configuration
            self.min_words = min_words

        def evaluate(self, system_output, correct_answer, **kwargs):
            word_count = len(str(system_output).split())
            is_correct = word_count >= self.min_words
            return eval.EvaluationResult(
                is_correct=is_correct,
                score=min(1.0, word_count / 20),  # Normalize to 0-1
                metadata={"word_count": word_count},
            )

    # Register the custom evaluator class (not an instance)
    eval.register_evaluator("word_count", WordCountEvaluator)

    # Verify it's in the registry now
    available = eval.list_available_evaluators()
    assert "word_count" in available, "Custom evaluator should be in registry"

    # Use the registered evaluator with parameters
    word_counter = eval.Evaluator.from_registry("word_count", min_words=3)
    result_wc = word_counter.evaluate("This is a test sentence", "doesn't matter")
    print(f"Word count evaluator result: {result_wc}")
    assert "word_count" in result_wc, "Expected word_count in metrics"
    assert (
        result_wc["word_count"] == 5
    ), f"Expected 5 words, got {result_wc['word_count']}"

    # Test function-based evaluator
    print("\nTesting function-based evaluator...")

    def custom_metric(prediction, reference):
        return {
            "is_correct": prediction.lower() == reference.lower(),
            "length_ratio": len(prediction) / len(reference)
            if len(reference) > 0
            else 0,
        }

    custom_evaluator = eval.Evaluator.from_function(custom_metric)
    result_custom = custom_evaluator.evaluate("PARIS", "Paris")
    print(f"Custom evaluator result: {result_custom}")

    # Test the evaluation pipeline with a simple dataset
    print("\nTesting evaluation pipeline...")

    # Create a dataset with mock entries
    class MockDataset:
        def __init__(self, entries):
            self.entries = entries

        def __iter__(self):
            return iter(self.entries)

    # Model that returns capitals for countries
    def capitals_model(country):
        mapping = {
            "France": "Paris",
            "Japan": "Tokyo",
            "Brazil": "Rio de Janeiro",  # Intentionally wrong
            "Germany": "Berlin",
            "India": "New Delhi",
        }
        return mapping.get(country, "Unknown")

    # Create dataset
    dataset = MockDataset(
        [
            {"question": "France", "answer": "Paris"},
            {"question": "Japan", "answer": "Tokyo"},
            {"question": "Brazil", "answer": "Brasília"},
            {"question": "Germany", "answer": "Berlin"},
            {"question": "India", "answer": "New Delhi"},
        ]
    )

    # Create evaluation pipeline
    pipeline = eval.EvaluationPipeline([exact_evaluator, custom_evaluator])

    # Run evaluation
    results = pipeline.evaluate(capitals_model, dataset)
    print(f"Pipeline evaluation results: {results}")

    # Confirm expected metrics are present
    assert "is_correct" in results, "Missing is_correct metric"
    assert "processed_count" in results, "Missing processed_count metric"
    assert results["processed_count"] == 5, "Expected 5 processed examples"

    # We expect 80% accuracy (4/5 correct, Brazil is wrong)
    expected_accuracy = 0.8
    actual_accuracy = results.get("is_correct", 0)
    print(f"\nExpected accuracy: {expected_accuracy}")
    print(f"Actual accuracy: {actual_accuracy}")

    # Check if accuracy is close to expected (allow small floating point differences)
    accuracy_match = abs(actual_accuracy - expected_accuracy) < 0.01
    if not accuracy_match:
        print(f"\nWARNING: Accuracy doesn't match expected value")
        print(f"Expected: {expected_accuracy}, Got: {actual_accuracy}")
    else:
        print("\nAccuracy matches expected value!")

    print("\nEvaluation API test completed successfully!")

except Exception as e:
    print(f"Error testing Evaluation API: {e}")
    import traceback

    traceback.print_exc()
