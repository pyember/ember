"""
MCQ Experiment Example: Multiple-Choice Question Evaluation Example

This simplified example demonstrates how to create and evaluate multiple-choice 
questions using different answer strategies, without requiring external API calls.

To run:
    uv run python src/ember/examples/data/mcq_experiment_example.py
"""

import argparse
import random
from typing import Dict, List, Tuple


class DatasetEntry:
    """A simple dataset entry with questions and choices."""

    def __init__(
        self, query: str, choices: Dict[str, str], metadata: Dict[str, str]
    ) -> None:
        """Initialize a dataset entry.

        Args:
            query: The question text
            choices: A dictionary of lettered choices
            metadata: Additional information
        """
        self.query = query
        self.choices = choices
        self.metadata = metadata


class EnsureValidChoiceOperator:
    """Validates and corrects answers to match the valid choices."""

    def __init__(self, name: str = "EnsureValidChoice") -> None:
        """Initialize the operator.

        Args:
            name: The name of this operator
        """
        self.name = name

    def __call__(
        self, query: str, partial_answer: str, choices: Dict[str, str]
    ) -> Dict[str, str]:
        """Process the input to ensure a valid choice is returned.

        Args:
            query: The question text
            partial_answer: The proposed answer that may need validation
            choices: The valid choices

        Returns:
            A dictionary with the final answer
        """
        # First check if the answer is already valid
        if partial_answer in choices:
            return {"final_answer": partial_answer}

        # If not valid, try to match to the most similar valid choice
        # In a real system, this would use an LLM, but here we'll do a simple mapping

        # First check if it's just a case mismatch
        for choice in choices:
            if choice.upper() == partial_answer.upper():
                return {"final_answer": choice}

        # Next, check if it's the text of an answer rather than the key
        for choice, text in choices.items():
            if partial_answer.lower() in text.lower():
                return {"final_answer": choice}

        # If all else fails, return the first choice as fallback
        if choices:
            return {"final_answer": next(iter(choices))}

        # If no choices at all, return empty
        return {"final_answer": ""}


class SingleModelBaseline:
    """A simple model that makes a choice based on fixed probabilities."""

    def __init__(self, name: str = "SingleModelBaseline") -> None:
        """Initialize the model.

        Args:
            name: The name of this model
        """
        self.name = name
        self.ensure_valid_choice = EnsureValidChoiceOperator()

    def __call__(self, query: str, choices: Dict[str, str]) -> Dict[str, str]:
        """Process a question with a single model approach.

        Args:
            query: The question text
            choices: The available choices

        Returns:
            A dictionary with the final answer
        """
        # In a real system, this would call a language model
        # Here we'll simulate it with a biased random choice

        # Generate a mock response based on keywords in the query
        if "capital" in query.lower():
            # For capital questions, prefer C as answer
            weights = {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1}
        elif "mammal" in query.lower() or "animal" in query.lower():
            # For biology questions, prefer D
            weights = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.7}
        elif any(keyword in query.lower() for keyword in ["math", "square", "number"]):
            # For math questions, prefer B
            weights = {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}
        else:
            # Otherwise equal chance
            weights = {k: 1.0 / len(choices) for k in choices}

        # Make a weighted random choice
        choices_list = list(weights.keys())
        weights_list = [weights[k] for k in choices_list]
        partial_answer = random.choices(choices_list, weights=weights_list, k=1)[0]

        # Ensure it's a valid choice
        result = self.ensure_valid_choice(
            query=query, partial_answer=partial_answer, choices=choices
        )
        return {"final_answer": result["final_answer"]}


class MultiModelEnsemble:
    """Simulates an ensemble of models making independent predictions."""

    def __init__(self, name: str = "MultiModelEnsemble", num_models: int = 3) -> None:
        """Initialize the ensemble.

        Args:
            name: The name of this ensemble
            num_models: The number of models in the ensemble
        """
        self.name = name
        self.num_models = num_models
        self.ensure_valid_choice = EnsureValidChoiceOperator()

    def __call__(self, query: str, choices: Dict[str, str]) -> Dict[str, str]:
        """Process a question with an ensemble approach.

        Args:
            query: The question text
            choices: The available choices

        Returns:
            A dictionary with the final answer
        """
        # Generate multiple responses simulating different models
        responses = []
        for _ in range(self.num_models):
            # Each "model" has slightly different preferences
            if "capital" in query.lower():
                weights = {
                    "A": 0.1,
                    "B": 0.1,
                    "C": 0.6 + random.uniform(-0.2, 0.2),
                    "D": 0.1,
                }
            elif "mammal" in query.lower() or "animal" in query.lower():
                weights = {
                    "A": 0.1,
                    "B": 0.1,
                    "C": 0.1,
                    "D": 0.6 + random.uniform(-0.2, 0.2),
                }
            elif any(
                keyword in query.lower() for keyword in ["math", "square", "number"]
            ):
                weights = {
                    "A": 0.1,
                    "B": 0.6 + random.uniform(-0.2, 0.2),
                    "C": 0.1,
                    "D": 0.1,
                }
            else:
                # Otherwise slightly different preferences per model
                weights = {
                    k: 1.0 / len(choices) + random.uniform(-0.1, 0.1) for k in choices
                }
                # Normalize weights
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}

            # Make a weighted random choice
            choices_list = list(weights.keys())
            weights_list = [weights[k] for k in choices_list]
            choice = random.choices(choices_list, weights=weights_list, k=1)[0]
            responses.append(choice)

        # Synthesize responses (use majority vote in this simplified version)
        # Count occurrences of each answer
        vote_counts = {}
        for response in responses:
            vote_counts[response] = vote_counts.get(response, 0) + 1

        # Find the majority answer
        if vote_counts:
            majority_answer = max(vote_counts.items(), key=lambda x: x[1])[0]
        else:
            majority_answer = next(iter(choices)) if choices else ""

        # Ensure it's a valid choice
        result = self.ensure_valid_choice(
            query=query, partial_answer=majority_answer, choices=choices
        )
        return {"final_answer": result["final_answer"]}


class VariedModelEnsemble:
    """Simulates different types of models in an ensemble."""

    def __init__(self, name: str = "VariedModelEnsemble") -> None:
        """Initialize the varied ensemble.

        Args:
            name: The name of this ensemble
        """
        self.name = name
        self.ensure_valid_choice = EnsureValidChoiceOperator()

    def __call__(self, query: str, choices: Dict[str, str]) -> Dict[str, str]:
        """Process a question with a varied model ensemble approach.

        Args:
            query: The question text
            choices: The available choices

        Returns:
            A dictionary with the final answer
        """
        # Simulate different model types with distinct characteristics
        model_types = [
            {"name": "fact_model", "strengths": ["capital", "country", "geography"]},
            {"name": "science_model", "strengths": ["mammal", "animal", "biology"]},
            {"name": "math_model", "strengths": ["math", "square", "number"]},
        ]

        # Generate responses from each model
        responses = []
        for model in model_types:
            # Calculate the confidence based on model's strengths
            confidence = 0.5  # baseline
            for strength in model["strengths"]:
                if strength in query.lower():
                    confidence += 0.2  # boost confidence for relevant questions

            # Generate answer based on model confidence and domain
            if model["name"] == "fact_model" and "capital" in query.lower():
                weights = {"A": 0.1, "B": 0.1, "C": 0.7 * confidence, "D": 0.1}
            elif model["name"] == "science_model" and (
                "mammal" in query.lower() or "animal" in query.lower()
            ):
                weights = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.7 * confidence}
            elif model["name"] == "math_model" and any(
                keyword in query.lower() for keyword in ["math", "square", "number"]
            ):
                weights = {"A": 0.1, "B": 0.7 * confidence, "C": 0.1, "D": 0.1}
            else:
                # Less confident outside its domain
                weights = {k: 1.0 / len(choices) for k in choices}

            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = {k: 1.0 / len(choices) for k in choices}

            # Make a weighted random choice
            choices_list = list(weights.keys())
            weights_list = [weights[k] for k in choices_list]
            choice = random.choices(choices_list, weights=weights_list, k=1)[0]
            responses.append(choice)

        # "Judge synthesis" - weight answers by confidence and domain relevance
        # This is simplified; in real systems this would be more complex
        if "capital" in query.lower() or "country" in query.lower():
            # Trust fact_model more for geography
            final_answer = responses[0]
        elif (
            "mammal" in query.lower()
            or "animal" in query.lower()
            or "biology" in query.lower()
        ):
            # Trust science_model more for biology
            final_answer = responses[1]
        elif any(keyword in query.lower() for keyword in ["math", "square", "number"]):
            # Trust math_model more for math
            final_answer = responses[2]
        else:
            # Otherwise take majority vote
            vote_counts = {}
            for response in responses:
                vote_counts[response] = vote_counts.get(response, 0) + 1

            if vote_counts:
                final_answer = max(vote_counts.items(), key=lambda x: x[1])[0]
            else:
                final_answer = next(iter(choices)) if choices else ""

        # Ensure it's a valid choice
        result = self.ensure_valid_choice(
            query=query, partial_answer=final_answer, choices=choices
        )
        return {"final_answer": result["final_answer"]}


def create_mock_dataset() -> List[DatasetEntry]:
    """Create a mock multiple-choice dataset in MMLU style."""
    return [
        DatasetEntry(
            query="What is the capital of France?",
            choices={"A": "Berlin", "B": "Madrid", "C": "Paris", "D": "Rome"},
            metadata={"correct_answer": "C", "subject": "Geography"},
        ),
        DatasetEntry(
            query="Which of these is a mammal?",
            choices={"A": "Shark", "B": "Snake", "C": "Eagle", "D": "Dolphin"},
            metadata={"correct_answer": "D", "subject": "Biology"},
        ),
        DatasetEntry(
            query="What is the square root of 144?",
            choices={"A": "10", "B": "12", "C": "14", "D": "16"},
            metadata={"correct_answer": "B", "subject": "Mathematics"},
        ),
        DatasetEntry(
            query="Who wrote 'Pride and Prejudice'?",
            choices={
                "A": "Jane Austen",
                "B": "Charles Dickens",
                "C": "Emily Brontë",
                "D": "Mark Twain",
            },
            metadata={"correct_answer": "A", "subject": "Literature"},
        ),
        DatasetEntry(
            query="Which planet is known as the Red Planet?",
            choices={"A": "Jupiter", "B": "Venus", "C": "Mars", "D": "Saturn"},
            metadata={"correct_answer": "C", "subject": "Astronomy"},
        ),
    ]


def score_entry(
    entry: DatasetEntry, pipeline_name: str, prediction: str
) -> Tuple[str, bool]:
    """Score a prediction against the correct answer.

    Args:
        entry: The dataset entry being evaluated
        pipeline_name: Name of the pipeline making the prediction
        prediction: The predicted answer

    Returns:
        A tuple with the pipeline name and whether the prediction was correct
    """
    correct_answer = entry.metadata.get("correct_answer", "").upper()
    prediction = prediction.upper()
    is_correct = prediction == correct_answer
    return (pipeline_name, is_correct)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MCQ Experiment Example demonstrating different answer strategies"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to process"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main() -> None:
    """Run the MCQ experiment example."""
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(args.seed)

    print("Ember MCQ Experiment Example")
    print("===========================")

    # Create dataset
    dataset = create_mock_dataset()
    dataset_size = min(args.num_samples, len(dataset))
    sample = dataset[:dataset_size]  # Take the first N entries

    # Create the pipelines
    baseline = SingleModelBaseline()
    ensemble = MultiModelEnsemble(num_models=3)
    varied = VariedModelEnsemble()

    # Track results
    results = {
        "SingleModelBaseline": {"correct": 0, "total": 0},
        "MultiModelEnsemble": {"correct": 0, "total": 0},
        "VariedModelEnsemble": {"correct": 0, "total": 0},
    }

    # Process each entry
    for i, entry in enumerate(sample):
        print(f"\nQuestion {i+1}: {entry.query}")
        print(f"Subject: {entry.metadata.get('subject', 'Unknown')}")

        # Show choices
        print("Choices:")
        correct = entry.metadata.get("correct_answer", "")
        for letter, text in entry.choices.items():
            is_correct = "✓" if letter == correct else " "
            print(f"  {letter}. {text} {is_correct}")

        # Process with each pipeline
        baseline_result = baseline(query=entry.query, choices=entry.choices)
        ensemble_result = ensemble(query=entry.query, choices=entry.choices)
        varied_result = varied(query=entry.query, choices=entry.choices)

        # Get predictions
        baseline_pred = baseline_result["final_answer"]
        ensemble_pred = ensemble_result["final_answer"]
        varied_pred = varied_result["final_answer"]

        # Score and update results
        for name, pred in [
            ("SingleModelBaseline", baseline_pred),
            ("MultiModelEnsemble", ensemble_pred),
            ("VariedModelEnsemble", varied_pred),
        ]:
            pipeline_name, is_correct = score_entry(entry, name, pred)
            results[pipeline_name]["total"] += 1
            if is_correct:
                results[pipeline_name]["correct"] += 1

            # Display prediction
            correct_str = "✓" if is_correct else "✗"
            print(f"{name} prediction: {pred} {correct_str}")

    # Show final results
    print("\nFinal Results:")
    print("-------------")
    header = f"{'Pipeline':<25} {'Accuracy':<10} {'Correct/Total'}"
    print(header)
    print("-" * len(header))

    for name, result in results.items():
        accuracy = (
            (result["correct"] / result["total"]) * 100 if result["total"] > 0 else 0
        )
        print(f"{name:<25} {accuracy:.2f}%      {result['correct']}/{result['total']}")

    print("\nNote: This example uses simulated models and predetermined outcomes.")
    print("In a real Ember pipeline, these would use actual language models.")


if __name__ == "__main__":
    main()
