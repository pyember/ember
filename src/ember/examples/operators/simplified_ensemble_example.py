"""Simplified Ensemble Operator Example

This example demonstrates an ensemble operator implementation that doesn't require
API access. It shows how to create, compose, and execute ensemble operators with proper
typing and simulation of multiple model responses.

To run:
    uv run python src/ember/examples/operators/simplified_ensemble_example.py
"""

import logging
import random
from typing import Any, ClassVar, List, Type

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field

###############################################################################
# Custom Input/Output Models
###############################################################################


class EnsembleInput(EmberModel):
    """Input for ensemble operator."""

    query: str = Field(description="The query to be processed")


class EnsembleOutput(EmberModel):
    """Output for ensemble operator."""

    responses: List[str] = Field(description="Multiple responses from the ensemble")
    final_answer: str = Field(description="The selected final answer")
    confidence: float = Field(description="Confidence in the final answer")


class EnsembleSpecification(Specification):
    """Specification for ensemble operator."""

    input_model: Type[EmberModel] = EnsembleInput
    structured_output: Type[EmberModel] = EnsembleOutput


###############################################################################
# Mock LM Module (Simulated Model API)
###############################################################################


class MockLMModule:
    """Simulates a language model without requiring API access."""

    def __init__(self, response_style: str = "normal", response_quality: str = "good"):
        """Initialize with configuration for response generation style.

        Args:
            response_style: The style of response to generate
            response_quality: The quality level of responses
        """
        self.response_style = response_style
        self.response_quality = response_quality

        # Some variety for each model
        self.style_phrases = {
            "concise": ["In short", "Briefly", "To summarize"],
            "detailed": [
                "In a comprehensive analysis",
                "Considering all factors",
                "In detail",
            ],
            "creative": ["Imagine", "Creatively speaking", "From a unique perspective"],
            "normal": ["", "Well,", "In response to your question"],
        }

        # Quality variants for simulation
        self.quality_modifiers = {
            "good": 0.9,  # High chance of correct answer
            "medium": 0.7,  # Moderate chance of correct answer
            "poor": 0.4,  # Low chance of correct answer
        }

        # Sample answers for common questions
        self.known_answers = {
            "capital of france": "Paris",
            "largest planet": "Jupiter",
            "closest star": "The Sun",
            "author of hamlet": "William Shakespeare",
            "tallest mountain": "Mount Everest",
            "number of planets": "Eight planets (or nine including Pluto)",
        }

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Generate a simulated response to the prompt.

        Args:
            prompt: The input text prompt
            **kwargs: Additional configuration parameters

        Returns:
            A simulated model response
        """
        # Select intro phrase based on style
        intro = random.choice(self.style_phrases.get(self.response_style, ["Well,"]))

        # Deterministically select the "correct" answer based on the prompt
        normalized_prompt = prompt.lower()
        answer = None

        # Check for known answers in the prompt
        for key, value in self.known_answers.items():
            if key in normalized_prompt:
                answer = value
                break

        # Default answer if no matching pattern
        if answer is None:
            answer = (
                "I don't have enough information to answer that question accurately."
            )

        # Based on quality, maybe give incorrect answer
        quality_score = self.quality_modifiers.get(self.response_quality, 0.7)
        if random.random() > quality_score:
            # Get a random wrong answer
            wrong_answers = list(self.known_answers.values())
            if answer in wrong_answers:
                wrong_answers.remove(answer)
            if wrong_answers:
                answer = random.choice(wrong_answers)

        # Add some style variations
        if self.response_style == "concise":
            return f"{intro}, {answer}"
        elif self.response_style == "detailed":
            return f"{intro}, I can tell you that {answer}. This is based on well-established facts."
        elif self.response_style == "creative":
            return f"{intro}, let's explore the answer: {answer} - which opens up interesting possibilities!"
        else:
            return f"{intro} {answer}"


###############################################################################
# Ensemble Operator
###############################################################################


class SimpleEnsembleOperator(Operator[EnsembleInput, EnsembleOutput]):
    """Operator that generates multiple responses using different simulated LM modules."""

    # Class-level specification declaration
    specification: ClassVar[Specification] = EnsembleSpecification()

    # Instance attributes
    lm_modules: List[MockLMModule]

    def __init__(self, num_units: int = 3) -> None:
        """Initialize with multiple simulated LM modules.

        Args:
            num_units: Number of simulated models to use
        """
        # Create a variety of LM modules with different styles and qualities
        styles = ["normal", "concise", "detailed", "creative"]
        qualities = ["good", "good", "medium", "poor"]  # Weighted toward good answers

        self.lm_modules = []
        for i in range(num_units):
            style = styles[i % len(styles)]
            quality = qualities[i % len(qualities)]
            self.lm_modules.append(
                MockLMModule(response_style=style, response_quality=quality)
            )

    def forward(self, *, inputs: EnsembleInput) -> EnsembleOutput:
        """Generate multiple responses for the input query.

        Args:
            inputs: The validated input containing a query

        Returns:
            Ensemble output containing multiple responses and aggregation
        """
        # Generate responses from all modules
        responses = [lm(inputs.query) for lm in self.lm_modules]

        # Simple aggregation - select most common response (like MostCommon operator)
        # Count occurrences of each response
        response_counts = {}
        for response in responses:
            response_counts[response] = response_counts.get(response, 0) + 1

        # Find the most common response
        max_count = 0
        final_answer = ""
        for response, count in response_counts.items():
            if count > max_count:
                max_count = count
                final_answer = response

        # If we couldn't find a most common, use the first response
        if not final_answer and responses:
            final_answer = responses[0]

        # Calculate confidence based on agreement ratio
        confidence = max_count / len(responses) if responses else 0.0

        return EnsembleOutput(
            responses=responses, final_answer=final_answer, confidence=confidence
        )


###############################################################################
# Main Demonstration
###############################################################################


def main() -> None:
    """Run the ensemble operator example."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== Simplified Ensemble Operator Example ===\n")

    # Create the ensemble
    ensemble = SimpleEnsembleOperator(num_units=5)

    # Example queries
    queries = [
        "What is the capital of France?",
        "What is the largest planet in our solar system?",
        "Who wrote Hamlet?",
        "What is the answer to an unknown question?",
    ]

    # Process all queries
    for query in queries:
        print(f"\nProcessing query: {query}")

        # Execute the operator with the query
        result = ensemble(inputs=EnsembleInput(query=query))

        # Display results
        print(f"Final answer: {result.final_answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"All responses ({len(result.responses)}):")
        for i, response in enumerate(result.responses, 1):
            print(f"  {i}. {response}")

    print("\nNote: This example demonstrates the ensemble pattern without requiring")
    print("      actual API access, using simulated model responses instead.")


if __name__ == "__main__":
    main()
