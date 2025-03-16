"""Example demonstrating the improved Ember Operators API.

This example shows how to create and compose operators using the simplified API,
demonstrating basic operators, ensemble patterns, and advanced composition techniques.
"""

from typing import List

from ember.api.operators import (
    Operator,
    EmberModel,
    Field,
    EnsembleOperator,
    MostCommonAnswerSelector,
    JudgeSynthesisOperator,
    Specification,
)


# Define input/output models
class QuestionInput(EmberModel):
    question: str = Field(..., description="The question to be answered")


class AnswerOutput(EmberModel):
    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(
        default=1.0, description="Confidence score for the answer"
    )


class MultipleAnswersOutput(EmberModel):
    answers: List[str] = Field(..., description="Multiple candidate answers")


# Basic operator example
class SimpleQuestionAnswerer(Operator[QuestionInput, AnswerOutput]):
    """A simple operator that answers questions using a language model."""

    specification: Specification[QuestionInput, AnswerOutput] = Specification(
        input_model=QuestionInput, output_model=AnswerOutput
    )

    model_name: str
    temperature: float

    def __init__(self, model_name: str, temperature: float):
        """Initialize the operator with model configuration.

        Args:
            model_name: The model to use for generation
            temperature: Sampling temperature for generation
        """
        self.model_name = model_name
        self.temperature = temperature

    def forward(self, inputs: QuestionInput) -> AnswerOutput:
        """Generate an answer to the input question.

        In a real implementation, this could call an lm.
        For demonstration, we're returning a static response.
        """
        # Simulate calling a language model
        answer = f"This is a response from {self.model_name}: The answer is 42."

        # Return structured outputyou
        return AnswerOutput(answer=answer, confidence=0.95)


# Diversification operator
class DiverseAnswerGenerator(Operator[QuestionInput, MultipleAnswersOutput]):
    """Generates multiple diverse answers to a question."""

    def __init__(self, prefixes: List[str], model_name: str = "gpt-4"):
        """Initialize with different prefixes to guide diverse responses.

        Args:
            prefixes: Different framing instructions to get diverse answers
            model_name: The model to use for generation
        """
        self.prefixes = prefixes
        self.model_name = model_name

    def forward(self, inputs: QuestionInput) -> MultipleAnswersOutput:
        """Generate multiple diverse answers using different prefixes."""
        answers = []

        for prefix in self.prefixes:
            # In real implementation, add prefix to prompt and call model
            answer = f"{prefix} answer for '{inputs.question}'"
            answers.append(answer)

        return MultipleAnswersOutput(answers=answers)


def main():
    """Run the example pipeline to demonstrate operator composition."""

    # Create a question input
    question = QuestionInput(question="What is the meaning of life?")

    print(f"Question: {question.question}\n")

    # 1. Simple operator
    simple_answerer = SimpleQuestionAnswerer(model_name="gpt-4")
    result1 = simple_answerer(question)
    print("1. Simple Operator:")
    print(f"   Answer: {result1.answer}")
    print(f"   Confidence: {result1.confidence}\n")

    # 2. Ensemble of operators
    ensemble = EnsembleOperator(
        operators=[
            SimpleQuestionAnswerer(model_name="gpt-4"),
            SimpleQuestionAnswerer(model_name="claude-3"),
            SimpleQuestionAnswerer(model_name="llama-3"),
        ]
    )
    result2 = ensemble(question)
    print("2. Ensemble Operator:")
    for i, output in enumerate(result2.outputs):
        print(f"   Model {i+1}: {output.answer}")
    print()

    # 3. Ensemble with answer selection
    pipeline = MostCommonAnswerSelector(operator=ensemble)
    result3 = pipeline(question)
    print("3. Ensemble with Most Common Answer Selector:")
    print(f"   Selected Answer: {result3.answer}\n")

    # 4. Diverse answers with synthesis
    diverse_generator = DiverseAnswerGenerator(
        prefixes=[
            "Scientific perspective:",
            "Philosophical perspective:",
            "Humorous perspective:",
        ]
    )

    synthesizer = JudgeSynthesisOperator(
        operator=diverse_generator,
        prompt_template="Synthesize the following answers: {answers}",
    )

    result4 = synthesizer(question)
    print("4. Diverse Answers with Synthesis:")
    print(f"   Synthesized Answer: {result4.answer}\n")


if __name__ == "__main__":
    main()
