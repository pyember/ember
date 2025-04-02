"""Example demonstrating the improved Ember Operators API with graceful fallbacks.

This example shows how to create and compose operators using the simplified API,
demonstrating basic operators, ensemble patterns, and advanced composition techniques.
It follows the latest API patterns and includes proper error handling for when
API keys are not available.

Run this example with:
```bash
uv run python -m src.ember.examples.integration.api_operators_example
```
"""

import logging
from typing import ClassVar, List

from ember.api.models import (
    ModelEnum,
    get_model_service,
    get_registry,
)
from ember.api.operators import (
    EnsembleOperator,
    Field,
    JudgeSynthesisOperator,
    MostCommonAnswerSelector,
    Operator,
    Specification,
)
from ember.core.registry.model.model_module.lm import LMModule
from ember.core.types.ember_model import EmberModel


# Define input/output models
class QuestionInput(EmberModel):
    """Input model for question answering."""

    question: str = Field(..., description="The question to be answered")


class AnswerOutput(EmberModel):
    """Output model for question answering."""

    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(
        default=1.0, description="Confidence score for the answer"
    )


class MultipleAnswersOutput(EmberModel):
    """Output model with multiple candidate answers."""

    answers: List[str] = Field(..., description="Multiple candidate answers")


# Helper class for simulating models when API keys aren't available
class MockLMModule:
    """Simulates a language model without requiring API access."""

    def __init__(self, model_name: str, temperature: float = 0.7):
        """Initialize mock LM with model name and temperature.

        Args:
            model_name: Name of the model to simulate
            temperature: Simulated temperature parameter
        """
        self.model_name = model_name
        self.temperature = temperature
        # Dictionary of canned responses
        self.canned_responses = {
            "What is the meaning of life?": f"This is a simulated response from {model_name}: The answer is 42."
        }

    def __call__(self, *, prompt: str) -> str:
        """Generate a simulated response based on the prompt.

        Args:
            prompt: The input prompt text

        Returns:
            A simulated model response
        """
        # Check for synthesis prompts
        if "synthesize" in prompt.lower() or "multiple advisors" in prompt.lower():
            return (
                "Reasoning: All three perspectives agree on the value '42' as the answer to the meaning of life, "
                "though they approach it from different angles. The scientific perspective presents it as a universal constant, "
                "the philosophical perspective frames it as an existential truth, and the humorous perspective references "
                "Douglas Adams' 'Hitchhiker's Guide to the Galaxy'. This convergence across different domains suggests '42' "
                "is indeed the most comprehensive answer.\n\n"
                "Final Answer: The meaning of life is 42."
            )

        # Check for prompt prefixes for diverse answers
        for prefix in [
            "Scientific perspective:",
            "Philosophical perspective:",
            "Humorous perspective:",
        ]:
            if prefix in prompt:
                if "Scientific" in prefix:
                    return f"From a {prefix} The mathematical constant 42 appears in various equations describing the universe."
                elif "Philosophical" in prefix:
                    return f"From a {prefix} The meaning of life is to find your own purpose and create meaning through authentic choices."
                elif "Humorous" in prefix:
                    return f"From a {prefix} According to the Hitchhiker's Guide to the Galaxy, the answer is definitively 42."

        # Check for known questions
        for question, answer in self.canned_responses.items():
            if question.lower() in prompt.lower():
                return answer

        # Generic response
        return f"Simulated response from {self.model_name}: I don't have a specific answer for this question."


class ModelProvider:
    """Helper class to provide real or mock models based on availability."""

    @staticmethod
    def get_lm_module(model_enum: ModelEnum, temperature: float = 0.7) -> LMModule:
        """Get a real or mocked LM module based on model availability.

        Args:
            model_enum: The model enum to get
            temperature: Temperature setting for the model

        Returns:
            Either a real LM module or a mock LM module
        """
        registry = get_registry()

        # Try to get the real model
        try:
            if hasattr(model_enum, "value") and registry.is_registered(
                model_enum.value
            ):
                model_service = get_model_service()
                model = model_service.get_model(model_enum.value)
                model.temperature = temperature
                return model
        except Exception as e:
            logging.warning(f"Could not load model {model_enum}: {str(e)}")

        # Fall back to mock implementation
        model_name = model_enum.name if hasattr(model_enum, "name") else str(model_enum)
        logging.info(f"Using mock implementation for {model_name}")
        return MockLMModule(model_name=model_name, temperature=temperature)


# Basic operator example
class SimpleQuestionAnswerer(Operator[QuestionInput, AnswerOutput]):
    """A simple operator that answers questions using a language model."""

    specification: ClassVar[Specification[QuestionInput, AnswerOutput]] = Specification(
        input_model=QuestionInput, structured_output=AnswerOutput
    )

    # Field declarations
    lm_module: LMModule

    def __init__(self, *, model_enum: ModelEnum, temperature: float = 0.7):
        """Initialize the operator with model configuration.

        Args:
            model_enum: The model enum to use
            temperature: Sampling temperature for generation
        """
        self.lm_module = ModelProvider.get_lm_module(model_enum, temperature)

    def forward(self, *, inputs: QuestionInput) -> AnswerOutput:
        """Generate an answer to the input question.

        Args:
            inputs: The question input model

        Returns:
            Structured answer output
        """
        # Call the language model with the question
        response = self.lm_module(prompt=inputs.question)

        # Extract text content from response if it's not already a string
        if hasattr(response, "data"):
            response_text = response.data
        else:
            response_text = str(response)

        # Return structured output
        return AnswerOutput(answer=response_text, confidence=0.95)


# Diversification operator
class DiverseAnswerGenerator(Operator[QuestionInput, MultipleAnswersOutput]):
    """Generates multiple diverse answers to a question."""

    specification: ClassVar[
        Specification[QuestionInput, MultipleAnswersOutput]
    ] = Specification(
        input_model=QuestionInput, structured_output=MultipleAnswersOutput
    )

    # Field declarations
    prefixes: List[str]
    lm_module: LMModule

    def __init__(self, *, prefixes: List[str], model_enum: ModelEnum = ModelEnum.gpt_4):
        """Initialize with different prefixes to guide diverse responses.

        Args:
            prefixes: Different framing instructions to get diverse answers
            model_enum: The model enum to use
        """
        self.prefixes = prefixes
        self.lm_module = ModelProvider.get_lm_module(model_enum)

    def forward(self, *, inputs: QuestionInput) -> MultipleAnswersOutput:
        """Generate multiple diverse answers using different prefixes.

        Args:
            inputs: The question input

        Returns:
            Multiple answers output
        """
        answers = []

        for prefix in self.prefixes:
            # Prepend the prefix to the question
            prompt = f"{prefix} {inputs.question}"
            response = self.lm_module(prompt=prompt)

            # Extract text content from response if it's not already a string
            if hasattr(response, "data"):
                response_text = response.data
            else:
                response_text = str(response)

            answers.append(response_text)

        return MultipleAnswersOutput(answers=answers)


def main():
    """Run the example pipeline to demonstrate operator composition."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check if API keys are available
    registry = get_registry()
    if not registry.list_models():
        logging.warning(
            "No models were discovered. This example will use mock models instead of real ones."
        )
        logging.warning(
            "To use real models, set one of these environment variables: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
        )

    # Create a question input
    question = QuestionInput(question="What is the meaning of life?")

    print(f"Question: {question.question}\n")

    # 1. Simple operator with GPT-4
    simple_answerer = SimpleQuestionAnswerer(model_enum=ModelEnum.gpt_4)
    result1 = simple_answerer(inputs=question)
    print("1. Simple Operator:")
    print(f"   Answer: {result1.answer}")
    print(f"   Confidence: {result1.confidence}\n")

    # Create a custom ensemble operator that handles response objects
    class CustomEnsembleOperator(EnsembleOperator):
        def forward(self, *, inputs: dict) -> dict:
            # Call the underlying models with the query
            raw_responses = [lm(prompt=inputs["query"]) for lm in self.lm_modules]

            # Process responses to extract text
            processed_responses = []
            for response in raw_responses:
                if hasattr(response, "data"):
                    processed_responses.append(response.data)
                else:
                    processed_responses.append(str(response))

            # Return the processed responses
            return {"responses": processed_responses}

    # Using fully typed model enums for clarity
    ensemble = CustomEnsembleOperator(
        lm_modules=[
            ModelProvider.get_lm_module(ModelEnum.gpt_4),
            ModelProvider.get_lm_module(ModelEnum.claude_3_5_sonnet),
            ModelProvider.get_lm_module(ModelEnum.gemini_1_5_pro),
        ]
    )

    result2 = ensemble(inputs={"query": question.question})

    print("2. Ensemble Operator:")
    for i, response in enumerate(result2["responses"], 1):
        print(f"   Model {i}: {response}")
    print()

    # 3. Ensemble with answer selection
    # Demonstrating direct invocation of the MostCommonAnswerSelector
    result3 = MostCommonAnswerSelector()(inputs={"responses": result2["responses"]})
    print("3. Ensemble with Most Common Answer Selector:")
    print(f"   Selected Answer: {result3['final_answer']}\n")

    # 4. Diverse answers with synthesis
    diverse_generator = DiverseAnswerGenerator(
        prefixes=[
            "Scientific perspective:",
            "Philosophical perspective:",
            "Humorous perspective:",
        ],
        model_enum=ModelEnum.gpt_4,
    )

    # Generate diverse answers
    diverse_results = diverse_generator(inputs=question)

    # Create a custom JudgeSynthesisOperator that handles response objects
    class CustomJudgeSynthesisOperator(JudgeSynthesisOperator):
        def forward(self, *, inputs: dict) -> dict:
            # Prepare synthesizer prompt
            question = inputs["query"]
            responses = inputs["responses"]

            # Build the prompt
            prompt = (
                f"I need you to synthesize the following perspectives on this question: '{question}'\n\n"
                "Multiple advisors have provided their views:\n\n"
            )

            for i, response in enumerate(responses, 1):
                prompt += f"Advisor {i}: {response}\n\n"

            prompt += (
                "Based on these perspectives, please provide:\n"
                "1. Your reasoning process, synthesizing the different viewpoints\n"
                "2. A final, balanced answer that represents the best synthesis\n\n"
                "Format your response like this:\n"
                "Reasoning: [your reasoning here]\n\n"
                "Final Answer: [your final answer here]"
            )

            # Get the response
            response = self.lm_module(prompt=prompt)

            # Extract the text from the response if it's not already a string
            if hasattr(response, "data"):
                response_text = response.data
            else:
                response_text = str(response)

            # Extract reasoning and final answer
            reasoning = ""
            final_answer = ""

            if "Reasoning:" in response_text and "Final Answer:" in response_text:
                parts = response_text.split("Final Answer:")
                if len(parts) >= 2:
                    reasoning_part = parts[0]
                    if "Reasoning:" in reasoning_part:
                        reasoning = reasoning_part.split("Reasoning:", 1)[1].strip()
                    final_answer = parts[1].strip()
            else:
                # Fallback if the expected format isn't found
                reasoning = "Could not extract reasoning"
                final_answer = response_text

            return {"reasoning": reasoning, "final_answer": final_answer}

    # Use the judge synthesis operator to synthesize the answers
    judge_lm = ModelProvider.get_lm_module(ModelEnum.claude_3_5_sonnet)
    synthesizer = CustomJudgeSynthesisOperator(lm_module=judge_lm)

    # Prepare the input for the synthesizer
    synthesis_input = {"query": question.question, "responses": diverse_results.answers}

    # Run the synthesizer
    result4 = synthesizer(inputs=synthesis_input)

    print("4. Diverse Answers with Synthesis:")
    print(f"   Synthesized Answer: {result4['final_answer']}")
    print(f"   Reasoning: {result4['reasoning']}\n")


if __name__ == "__main__":
    main()
