"""DSPy optimization examples with Ember backend."""

from typing import List, Tuple

import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

from ember.integrations.dspy import EmberLM


def create_math_dataset() -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Create a simple math word problem dataset."""
    # Training examples
    trainset = [
        dspy.Example(
            question="Sarah has 5 apples. She buys 3 more apples. How many apples does she have?",
            answer="8",
        ).with_inputs("question"),
        dspy.Example(
            question="A store has 20 books. They sell 7 books. How many books are left?",
            answer="13",
        ).with_inputs("question"),
        dspy.Example(
            question="Tom runs 4 miles every day. How many miles does he run in a week?",
            answer="28",
        ).with_inputs("question"),
        dspy.Example(
            question="A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?",
            answer="2",
        ).with_inputs("question"),
    ]

    # Test examples
    testset = [
        dspy.Example(
            question="Jenny has 12 marbles. She gives 4 to her friend. How many marbles does she have left?",
            answer="8",
        ).with_inputs("question"),
        dspy.Example(
            question="A farmer has 15 chickens. He buys 8 more. How many chickens does he have in total?",
            answer="23",
        ).with_inputs("question"),
    ]

    return trainset, testset


def example_bootstrap_optimization():
    """Optimize a math solver using bootstrap few-shot learning."""
    print("=== Bootstrap Few-Shot Optimization ===")

    # Initialize with a smaller model for optimization
    ember_lm = EmberLM(model="gpt-3.5-turbo", temperature=0.1)
    dspy.configure(lm=ember_lm)

    # Create dataset
    trainset, testset = create_math_dataset()

    # Define the program to optimize
    class MathSolver(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.solve(question=question)

    # Define evaluation metric
    def math_metric(example, prediction, trace=None):
        # Extract numeric answer from prediction
        try:
            pred_answer = prediction.answer.strip()
            # Handle cases where model explains before giving number
            if pred_answer.isdigit():
                pred_num = int(pred_answer)
            else:
                # Try to extract last number in response
                import re

                numbers = re.findall(r"\d+", pred_answer)
                pred_num = int(numbers[-1]) if numbers else -1

            true_num = int(example.answer)
            return pred_num == true_num
        except:
            return False

    # Create and compile program
    solver = MathSolver()

    print("Before optimization:")
    for example in testset[:2]:
        prediction = solver(question=example.question)
        correct = math_metric(example, prediction)
        print(f"Q: {example.question}")
        print(f"Predicted: {prediction.answer}")
        print(f"Expected: {example.answer}")
        print(f"Correct: {correct}\n")

    # Optimize with bootstrap few-shot
    print("Optimizing...")
    optimizer = BootstrapFewShot(metric=math_metric, max_bootstrapped_demos=4)
    optimized_solver = optimizer.compile(solver, trainset=trainset)

    print("\nAfter optimization:")
    for example in testset[:2]:
        prediction = optimized_solver(question=example.question)
        correct = math_metric(example, prediction)
        print(f"Q: {example.question}")
        print(f"Predicted: {prediction.answer}")
        print(f"Expected: {example.answer}")
        print(f"Correct: {correct}\n")


def example_signature_optimization():
    """Optimize instruction prompts for better performance."""
    print("=== Signature Optimization Example ===")

    # Use Claude for optimization
    ember_lm = EmberLM(model="claude-3-haiku-20240307", temperature=0.2)
    dspy.configure(lm=ember_lm)

    # Create sentiment analysis dataset
    trainset = [
        dspy.Example(text="This movie was absolutely fantastic!", sentiment="positive").with_inputs(
            "text"
        ),
        dspy.Example(
            text="Terrible service, would not recommend.", sentiment="negative"
        ).with_inputs("text"),
        dspy.Example(text="It was okay, nothing special.", sentiment="neutral").with_inputs("text"),
        dspy.Example(text="Best purchase I've ever made!", sentiment="positive").with_inputs(
            "text"
        ),
        dspy.Example(text="Complete waste of money.", sentiment="negative").with_inputs("text"),
    ]

    testset = [
        dspy.Example(text="Exceeded all my expectations!", sentiment="positive").with_inputs(
            "text"
        ),
        dspy.Example(text="Not worth the price.", sentiment="negative").with_inputs("text"),
        dspy.Example(text="Average product, does the job.", sentiment="neutral").with_inputs(
            "text"
        ),
    ]

    # Base program
    class SentimentClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.classify = dspy.Predict("text -> sentiment")

        def forward(self, text):
            return self.classify(text=text)

    # Metric
    def sentiment_metric(example, prediction, trace=None):
        pred_sentiment = prediction.sentiment.lower().strip()
        true_sentiment = example.sentiment.lower().strip()

        # Handle variations
        if "positive" in pred_sentiment:
            pred_sentiment = "positive"
        elif "negative" in pred_sentiment:
            pred_sentiment = "negative"
        elif "neutral" in pred_sentiment:
            pred_sentiment = "neutral"

        return pred_sentiment == true_sentiment

    # Test before optimization
    classifier = SentimentClassifier()
    print("Performance before optimization:")
    correct = sum(sentiment_metric(ex, classifier(text=ex.text)) for ex in testset)
    print(f"Accuracy: {correct}/{len(testset)} = {correct / len(testset) * 100:.1f}%\n")

    # Optimize with random search
    print("Running optimization with random search...")
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=sentiment_metric, max_bootstrapped_demos=3, num_candidate_programs=5
    )
    optimized_classifier = optimizer.compile(classifier, trainset=trainset)

    # Test after optimization
    print("\nPerformance after optimization:")
    results = []
    for ex in testset:
        pred = optimized_classifier(text=ex.text)
        is_correct = sentiment_metric(ex, pred)
        results.append(is_correct)
        print(f"Text: {ex.text}")
        print(f"Predicted: {pred.sentiment}, Expected: {ex.sentiment}, Correct: {is_correct}")

    accuracy = sum(results) / len(results)
    print(f"\nFinal Accuracy: {accuracy * 100:.1f}%")


def example_multi_model_ensemble_optimization():
    """Optimize an ensemble using multiple Ember models."""
    print("=== Multi-Model Ensemble Optimization ===")

    # Create training data for a classification task
    categories = ["Technology", "Sports", "Politics", "Entertainment"]

    trainset = [
        dspy.Example(
            text="The new iPhone features an improved camera and longer battery life.",
            category="Technology",
        ).with_inputs("text"),
        dspy.Example(
            text="The Lakers won the championship after a thrilling final game.",
            category="Sports",
        ).with_inputs("text"),
        dspy.Example(
            text="The senator announced new legislation on healthcare reform.",
            category="Politics",
        ).with_inputs("text"),
        dspy.Example(
            text="The movie won three Academy Awards including Best Picture.",
            category="Entertainment",
        ).with_inputs("text"),
        # Add more examples...
    ]

    testset = [
        dspy.Example(
            text="Apple announced their latest MacBook with M3 chip.",
            category="Technology",
        ).with_inputs("text"),
        dspy.Example(
            text="The quarterback threw for 400 yards in the playoff game.",
            category="Sports",
        ).with_inputs("text"),
    ]

    class EnsembleClassifier(dspy.Module):
        def __init__(self, models: List[str]):
            super().__init__()
            self.models = models
            self.classifiers = [dspy.Predict("text -> category") for _ in models]

        def forward(self, text):
            predictions = []

            # Get predictions from each model
            for model_name, classifier in zip(self.models, self.classifiers, strict=False):
                # Switch to specific model
                ember_lm = EmberLM(model=model_name, temperature=0.1)
                with dspy.context(lm=ember_lm):
                    pred = classifier(text=text)
                    predictions.append(pred.category)

            # Simple majority voting
            from collections import Counter

            vote_counts = Counter(predictions)
            majority_category = vote_counts.most_common(1)[0][0]

            return dspy.Prediction(
                category=majority_category,
                votes=dict(vote_counts),
                model_predictions=dict(zip(self.models, predictions, strict=False)),
            )

    # Define metric
    def category_metric(example, prediction, trace=None):
        return prediction.category.strip() == example.category.strip()

    # Test different model combinations
    model_combinations = [
        ["gpt-3.5-turbo"],  # Single model baseline
        ["gpt-3.5-turbo", "claude-3-haiku-20240307"],  # Two model ensemble
        ["gpt-3.5-turbo", "claude-3-haiku-20240307", "gpt-4"],  # Three model ensemble
    ]

    for models in model_combinations:
        print(f"\nTesting with models: {models}")

        # Configure default model
        ember_lm = EmberLM(model=models[0])
        dspy.configure(lm=ember_lm)

        # Create and test classifier
        classifier = EnsembleClassifier(models)

        # Run on test set
        correct = 0
        for ex in testset:
            pred = classifier(text=ex.text)
            is_correct = category_metric(ex, pred)
            correct += is_correct
            print(f"Text: {ex.text[:50]}...")
            print(f"Predictions by model: {pred.model_predictions}")
            print(f"Final prediction: {pred.category} (correct: {is_correct})")

        accuracy = correct / len(testset)
        print(f"Accuracy: {accuracy * 100:.1f}%")

    # Get usage metrics for cost analysis
    print("\n=== Cost Analysis ===")
    print("Total usage across all models:")
    metrics = ember_lm.get_usage_metrics()
    print(f"Total calls: {metrics.get('total_calls', 0)}")
    print(f"Total tokens: {metrics.get('total_tokens', 0)}")
    print(f"Total cost: ${metrics.get('total_cost', 0):.4f}")


if __name__ == "__main__":
    # Run optimization examples
    example_bootstrap_optimization()
    print("\n" + "=" * 60 + "\n")

    example_signature_optimization()
    print("\n" + "=" * 60 + "\n")

    example_multi_model_ensemble_optimization()
