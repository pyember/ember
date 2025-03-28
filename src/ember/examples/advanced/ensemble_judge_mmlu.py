"""
Advanced example demonstrating a varied ensemble operator with a judge for MMLU evaluation.

This example:
1. Creates a varied ensemble operator using multiple models with different configurations
2. Feeds ensemble responses to a judge operator
3. Evaluates on MMLU dataset with selectable subjects
4. Compares to a baseline single model operator
5. Optimizes execution using XCS transforms
6. Visualizes performance results

Usage:
    # Basic usage (2 subjects, 2 samples each, 2 models)
    python -m ember.examples.advanced.ensemble_judge_mmlu
    
    # Custom configuration through environment variables
    SAMPLE_SIZE=5 MMLU_SUBJECTS="high_school_mathematics,philosophy" MODEL_COUNT=3 \
        python -m ember.examples.advanced.ensemble_judge_mmlu
        
    # Quick test with minimal API calls
    SAMPLE_SIZE=1 MMLU_SUBJECTS="high_school_mathematics" MODEL_COUNT=1 \
        python -m ember.examples.advanced.ensemble_judge_mmlu
        
Environment variables:
    SAMPLE_SIZE: Number of samples per subject (default: 2)
    MMLU_SUBJECTS: Comma-separated list of subjects (default: first 2 subjects)
    MODEL_COUNT: Number of models to use in the ensemble (default: 2)
"""

import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import necessary modules
from ember.api import models, operators, Dataset, DatasetBuilder
from ember.api.operators import Operator
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base.operator_base import Specification
from ember.core.registry.specification.specification import (
    Specification as CoreSpecification,
)
from ember.core.types.ember_model import EmberModel
from ember.core.utils.data.base.models import DatasetEntry
from ember.xcs import jit, pmap
from ember.xcs.engine.execution_options import execution_options

# Set up console for rich output
console = Console()

# Available MMLU subjects for evaluation
MMLU_SUBJECTS = [
    "high_school_mathematics",
    "high_school_computer_science",
    "high_school_physics",
    "professional_medicine",
    "college_biology",
    "philosophy",
]


# Define input/output data structures using EmberModel for compatibility with XCS
class MCQInput(EmberModel):
    """Input for multiple-choice question evaluation."""

    question: str
    choices: Dict[str, str]


class MCQOutput(EmberModel):
    """Output for multiple-choice question evaluation."""

    answer: str
    reasoning: str = ""
    confidence: float = 0.0


class EnsembleJudgeInput(EmberModel):
    """Input for the judge operator."""

    question: str
    choices: Dict[str, str]
    candidate_responses: List[MCQOutput]


class EnsembleJudgeOutput(EmberModel):
    """Output for the judge operator."""

    selected_answer: str
    confidence: float
    justification: str


class MMLUDataset:
    """Dataset loader for MMLU evaluations with subject selection.

    Uses the Ember dataset API to load MMLU datasets by subject.
    """

    def __init__(self, subject: str = "high_school_mathematics"):
        """Initialize MMLU dataset loader with specified subject.

        Args:
            subject: The MMLU subject to load (config_name in MMLU)
        """
        self.subject = subject

    def load(self, max_samples: int = 10) -> List[Dict[str, Any]]:
        """Load samples from the specified MMLU subject.

        Args:
            max_samples: Maximum number of samples to load

        Returns:
            List of question dictionaries formatted for operators
        """
        try:
            # Use the standard DatasetBuilder pattern to load MMLU data
            dataset = (
                DatasetBuilder()
                .from_registry("mmlu")
                .subset(self.subject)  # MMLU uses subset as the subject name
                .split("test")  # Test split for evaluation
                .sample(max_samples)  # Limit number of samples
                .build()
            )

            # Convert DatasetEntry objects to our expected format
            result = []
            for entry in dataset.entries:
                result.append(
                    {
                        "question": entry.query,
                        "choices": entry.choices,
                        "answer": entry.metadata.get("correct_answer", ""),
                    }
                )
            return result

        except Exception as e:
            # Fallback to mock data if dataset loading fails
            console.print(f"[yellow]Using mock data: {str(e)}[/yellow]")
            return self._get_mock_data(max_samples)

    def _get_mock_data(self, max_samples: int) -> List[Dict[str, Any]]:
        """Get mock data for testing when real dataset cannot be loaded.

        Args:
            max_samples: Maximum number of samples to return

        Returns:
            List of mock question dictionaries
        """
        # Mock data by subject
        mock_datasets = {
            "high_school_mathematics": [
                {
                    "question": "What is the derivative of f(x) = x^2?",
                    "choices": {
                        "A": "f'(x) = x",
                        "B": "f'(x) = 2x",
                        "C": "f'(x) = 2",
                        "D": "f'(x) = x^2",
                    },
                    "answer": "B",
                },
                {
                    "question": "Solve for x: 2x + 5 = 13",
                    "choices": {"A": "x = 3", "B": "x = 4", "C": "x = 5", "D": "x = 6"},
                    "answer": "B",
                },
                {
                    "question": "What is the area of a circle with radius 5?",
                    "choices": {"A": "25π", "B": "10π", "C": "5π", "D": "100π"},
                    "answer": "A",
                },
            ],
            "high_school_physics": [
                {
                    "question": "What is the SI unit of force?",
                    "choices": {
                        "A": "Joule",
                        "B": "Newton",
                        "C": "Watt",
                        "D": "Pascal",
                    },
                    "answer": "B",
                },
                {
                    "question": "Which law states that energy cannot be created or destroyed?",
                    "choices": {
                        "A": "Newton's First Law",
                        "B": "Law of Conservation of Energy",
                        "C": "Ohm's Law",
                        "D": "Boyle's Law",
                    },
                    "answer": "B",
                },
            ],
            "philosophy": [
                {
                    "question": "Who wrote 'Critique of Pure Reason'?",
                    "choices": {
                        "A": "Friedrich Nietzsche",
                        "B": "Immanuel Kant",
                        "C": "René Descartes",
                        "D": "John Locke",
                    },
                    "answer": "B",
                },
                {
                    "question": "The statement 'I think, therefore I am' is associated with which philosopher?",
                    "choices": {
                        "A": "Socrates",
                        "B": "Aristotle",
                        "C": "René Descartes",
                        "D": "Plato",
                    },
                    "answer": "C",
                },
            ],
        }

        # Get the specified subject or default to mathematics
        data = mock_datasets.get(self.subject, mock_datasets["high_school_mathematics"])
        return data[:max_samples]


class BaselineMCQSpecification(CoreSpecification):
    """Specification for baseline MCQ operator."""

    # Define both input and output models explicitly
    input_model: Type[MCQInput] = MCQInput
    structured_output: Type[MCQOutput] = MCQOutput

    prompt_template: str = """Answer the following multiple-choice question:

Question: {question}

Options:
{choices}

First, think through this step-by-step to determine the correct answer.
Then, respond with just one of the option letters (A, B, C, or D).
Explain your reasoning after selecting your answer.

Format:
Answer: [Selected Option Letter]
Reasoning: [Your detailed explanation]
"""


class BaselineMCQOperator(Operator[MCQInput, MCQOutput]):
    """Baseline operator using a single model for MCQ evaluation."""

    specification: ClassVar[Specification] = BaselineMCQSpecification()
    model_name: str
    temperature: float
    max_tokens: int
    lm_module: LMModule

    def __init__(
        self,
        model_name: str = "anthropic:claude-3-haiku",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize the baseline operator with model configuration.

        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the LM module
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
            )
        )

    def forward(self, *, inputs: MCQInput) -> MCQOutput:
        """Process a multiple-choice question with the model.

        Args:
            inputs: Question and choices

        Returns:
            Model's answer and reasoning
        """
        # Pre-format choices as text for template insertion
        choices_text = "\n".join(
            [f"{key}. {value}" for key, value in inputs.choices.items()]
        )

        # Create a modified copy of inputs with the formatted choices
        template_vars = {
            "question": inputs.question,
            "choices": choices_text,  # Pre-formatted choices text passed as "choices"
        }

        # Fill in the template with standard Python format strings
        prompt = self.specification.prompt_template.format(**template_vars)

        # Get model response
        response = self.lm_module(prompt=prompt)

        # Parse the response to extract answer and reasoning
        answer, reasoning = self._parse_response(response, inputs.choices)

        return MCQOutput(
            answer=answer,
            reasoning=reasoning,
            confidence=1.0,  # Default confidence for baseline
        )

    def _parse_response(
        self, response: str, choices: Dict[str, str]
    ) -> Tuple[str, str]:
        """Parse the model response to extract answer and reasoning.

        Args:
            response: The full text response from the model
            choices: The available choices

        Returns:
            Tuple of (selected answer, reasoning)
        """
        if not response:
            return "Unknown", "No response"

        response = response.strip()
        answer_letter = None
        reasoning = response

        # Try to extract answer using "Answer:" format
        if "Answer:" in response:
            parts = response.split("Answer:", 1)
            if len(parts) > 1:
                answer_part = parts[1].strip().split("\n", 1)[0].strip()
                # Extract just the letter part
                for letter in choices:
                    if letter in answer_part:
                        answer_letter = letter
                        break

                # Extract reasoning if it's in the expected format
                if "Reasoning:" in response:
                    reasoning = response.split("Reasoning:", 1)[1].strip()
                else:
                    # If no explicit reasoning section, use everything after the answer
                    reasoning_parts = parts[1].strip().split("\n", 1)
                    reasoning = (
                        reasoning_parts[1].strip()
                        if len(reasoning_parts) > 1
                        else "No explicit reasoning provided."
                    )

        # If no answer found with "Answer:" format, try other approaches
        if not answer_letter:
            for letter in choices:
                if (
                    f"The answer is {letter}" in response
                    or f"answer is {letter}" in response
                ):
                    answer_letter = letter
                    break

        # If still no answer, look for letter at beginning of lines
        if not answer_letter:
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                for letter in choices:
                    if line.startswith(f"{letter}") or line.startswith(f"({letter})"):
                        answer_letter = letter
                        break
                if answer_letter:
                    break

        # Last resort: Just look for the first occurrence of any valid choice letter
        if not answer_letter:
            for letter in choices:
                if letter in response:
                    answer_letter = letter
                    break

        # Default to first choice if we couldn't extract an answer
        if not answer_letter and choices:
            answer_letter = next(iter(choices), None)

        # Get the full answer text
        answer = choices.get(answer_letter, "Unknown")

        return answer, reasoning


class VariedEnsembleSpecification(CoreSpecification):
    """Specification for varied ensemble MCQ operator."""

    # Explicit models for proper type handling
    input_model: Type[MCQInput] = MCQInput
    # We'll let the system handle the conversion of lists
    structured_output: Any = None

    # Different prompt templates for each model strategy
    prompt_templates: List[str] = [
        # Template for analytical approach
        """Analyze this multiple-choice question from a logical perspective:

Question: {question}

Options:
{choices}

Use logical analysis to determine the correct answer.
First, analyze each option systematically.
Then, select your answer and explain your reasoning.

Answer: [Option Letter]
Reasoning: [Explanation]
""",
        # Template for example-based approach
        """Answer this multiple-choice question by comparing with examples:

Question: {question}

Options:
{choices}

Think about similar examples or cases to help solve this.
Compare the options to known concepts or facts.
Select your answer and explain your reasoning.

Answer: [Option Letter]
Reasoning: [Explanation]
""",
        # Template for comprehensive approach
        """Consider all aspects of this multiple-choice question:

Question: {question}

Options:
{choices}

Evaluate each option thoroughly and eliminate incorrect choices.
Consider all relevant facts and concepts.
Select your answer and provide comprehensive reasoning.

Answer: [Option Letter]
Reasoning: [Explanation]
""",
    ]


class VariedEnsembleMCQOperator(Operator[MCQInput, List[MCQOutput]]):
    """Ensemble operator using multiple model configurations for MCQ evaluation.

    This operator implements a varied ensemble approach where multiple language models
    with different configurations evaluate the same question. The implementation is
    structured to enable automatic parallelization by JIT optimization.
    """

    specification: ClassVar[Specification] = VariedEnsembleSpecification()
    lm_modules: List[LMModule]

    def __init__(
        self,
        model_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the varied ensemble operator with multiple model configurations.

        Args:
            model_configs: List of model configuration dictionaries
        """
        if model_configs is None:
            # Default model configurations with different models and parameters
            model_configs = [
                {"model_name": "anthropic:claude-3-haiku", "temperature": 0.0},
                {"model_name": "anthropic:claude-3-haiku", "temperature": 0.7},
                {"model_name": "openai:gpt-3.5-turbo", "temperature": 0.0},
                {"model_name": "openai:gpt-4", "temperature": 0.0},
            ]

        # Create LM modules for each configuration
        self.lm_modules = []
        for config in model_configs:
            self.lm_modules.append(
                LMModule(
                    config=LMModuleConfig(
                        model_name=config["model_name"],
                        temperature=config["temperature"],
                        max_tokens=config.get("max_tokens", 1024),
                    )
                )
            )

    def _process_with_model(
        self, question: str, choices: Dict[str, str], lm_module: LMModule, template: str
    ) -> MCQOutput:
        """Process a question with a single model.

        This method is separated to make it clear to the JIT optimizer that
        each model inference can be parallelized.

        Args:
            question: The question text
            choices: Original choices dictionary
            lm_module: The language model module to use
            template: Prompt template to format

        Returns:
            Processed output with answer and reasoning
        """
        # Pre-format choices as text for template insertion
        choices_text = "\n".join([f"{key}. {value}" for key, value in choices.items()])

        # Create template variables with pre-formatted choices
        template_vars = {
            "question": question,
            "choices": choices_text,  # Pre-formatted choices text passed as "choices"
        }

        # Fill in the template with standard Python format strings
        prompt = template.format(**template_vars)

        # Get model response
        response = lm_module(prompt=prompt)

        # Parse the response
        answer, reasoning = self._parse_response(response, choices)

        return MCQOutput(
            answer=answer,
            reasoning=reasoning,
            confidence=0.8,  # Default confidence, could be improved with calibration
        )

    def forward(self, *, inputs: MCQInput) -> List[MCQOutput]:
        """Process a question with multiple models in the ensemble.

        This implementation is structured to allow the compiler to recognize
        the independent model calls, enabling automatic parallelization.

        Args:
            inputs: Question and choices

        Returns:
            List of answers and reasoning from each model
        """
        templates = self.specification.prompt_templates

        # Get responses from each model with different prompts
        responses = []
        for i, lm_module in enumerate(self.lm_modules):
            # Select a prompt template (cycling through available templates)
            template_idx = i % len(templates)
            template = templates[template_idx]

            # Process with this model - JIT optimizer can recognize these as independent operations
            responses.append(
                self._process_with_model(
                    question=inputs.question,
                    choices=inputs.choices,
                    lm_module=lm_module,
                    template=template,
                )
            )

        return responses

    def _parse_response(
        self, response: str, choices: Dict[str, str]
    ) -> Tuple[str, str]:
        """Parse the model response to extract answer and reasoning.

        Args:
            response: The full text response from the model
            choices: The available choices

        Returns:
            Tuple of (selected answer, reasoning)
        """
        if not response:
            return "Unknown", "No response"

        response = response.strip()
        answer_letter = None
        reasoning = response

        # Try to extract answer using "Answer:" format
        if "Answer:" in response:
            parts = response.split("Answer:", 1)
            if len(parts) > 1:
                answer_part = parts[1].strip().split("\n", 1)[0].strip()
                # Extract just the letter part
                for letter in choices:
                    if letter in answer_part:
                        answer_letter = letter
                        break

                # Extract reasoning if it's in the expected format
                if "Reasoning:" in response:
                    reasoning = response.split("Reasoning:", 1)[1].strip()
                else:
                    # If no explicit reasoning section, use everything after the answer
                    reasoning_parts = parts[1].strip().split("\n", 1)
                    reasoning = (
                        reasoning_parts[1].strip()
                        if len(reasoning_parts) > 1
                        else "No explicit reasoning provided."
                    )

        # If no answer found with "Answer:" format, try other approaches
        if not answer_letter:
            for letter in choices:
                if (
                    f"The answer is {letter}" in response
                    or f"answer is {letter}" in response
                ):
                    answer_letter = letter
                    break

        # Default to first choice if we couldn't extract an answer
        if not answer_letter and choices:
            answer_letter = next(iter(choices), None)

        # Get the full answer text
        answer = choices.get(answer_letter, "Unknown")

        return answer, reasoning


class JudgeOperatorSpecification(CoreSpecification):
    """Specification for MCQ judge operator."""

    # Explicit models for proper type handling
    input_model: Type[EnsembleJudgeInput] = EnsembleJudgeInput
    structured_output: Type[EnsembleJudgeOutput] = EnsembleJudgeOutput

    prompt_template: str = """You are a judge evaluating different candidate answers to a multiple-choice question.
Your task is to analyze the reasoning of each candidate and select the most accurate answer.

Question: {question}

Options:
{choices}

Candidate responses:
{candidate_responses}

Step 1: Analyze each candidate's reasoning objectively and identify strengths and weaknesses.
Step 2: Determine which reasoning is most sound and leads to the correct answer.
Step 3: Select the answer you believe is correct based on the strongest reasoning.
Step 4: Provide your justification for this selection.

Your response should follow this format:
Selected Answer: [selected answer text]
Confidence: [number between 0-1 representing certainty]
Justification: [your detailed reasoning justifying this selection]
"""


class JudgeOperator(Operator[EnsembleJudgeInput, EnsembleJudgeOutput]):
    """Judge operator that selects the best answer from ensemble responses."""

    specification: ClassVar[Specification] = JudgeOperatorSpecification()
    lm_module: LMModule

    def __init__(
        self,
        model_name: str = "anthropic:claude-3-sonnet",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize the judge operator with model configuration.

        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
        """
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
            )
        )

    def forward(self, *, inputs: EnsembleJudgeInput) -> EnsembleJudgeOutput:
        """Judge ensemble responses and select the best answer.

        Args:
            inputs: Question, choices, and candidate responses

        Returns:
            The selected answer with justification
        """
        # Format candidate responses
        candidate_responses_text = ""
        for i, response in enumerate(inputs.candidate_responses):
            candidate_responses_text += f"Candidate {i+1}:\n"
            candidate_responses_text += f"Answer: {response.answer}\n"
            candidate_responses_text += f"Reasoning: {response.reasoning}\n\n"

        # Pre-format choices as text for template insertion
        choices_text = "\n".join(
            [f"{key}. {value}" for key, value in inputs.choices.items()]
        )

        # Create template variables with pre-formatted choices
        template_vars = {
            "question": inputs.question,
            "choices": choices_text,  # Pre-formatted choices text passed as "choices"
            "candidate_responses": candidate_responses_text,
        }

        # Fill in the template with standard Python format strings
        prompt = self.specification.prompt_template.format(**template_vars)

        # Get model response
        response = self.lm_module(prompt=prompt)

        # Parse the judge's response
        selected_answer, confidence, justification = self._parse_judge_response(
            response, inputs.choices
        )

        return EnsembleJudgeOutput(
            selected_answer=selected_answer,
            confidence=confidence,
            justification=justification,
        )

    def _parse_judge_response(
        self, response: str, choices: Dict[str, str]
    ) -> Tuple[str, float, str]:
        """Parse the judge's response to extract the selected answer and justification.

        Args:
            response: The full text response from the judge
            choices: The available choices

        Returns:
            Tuple of (selected answer, confidence, justification)
        """
        if not response:
            return "Unknown", 0.5, "No response"

        response = response.strip()

        # Default values
        selected_answer = "Unknown"
        confidence = 0.5
        justification = "No justification provided."

        # Try to extract the selected answer using the expected format
        if "Selected Answer:" in response:
            parts = response.split("Selected Answer:", 1)
            if len(parts) > 1:
                answer_part = parts[1].split("\n", 1)[0].strip()

                # Check if the answer matches any of the choices
                for letter, choice_text in choices.items():
                    if letter in answer_part or choice_text in answer_part:
                        selected_answer = choice_text
                        break

        # Extract confidence if available
        if "Confidence:" in response:
            parts = response.split("Confidence:", 1)
            if len(parts) > 1:
                confidence_part = parts[1].split("\n", 1)[0].strip()
                try:
                    confidence_value = float(confidence_part)
                    if 0 <= confidence_value <= 1:
                        confidence = confidence_value
                except (ValueError, TypeError):
                    pass

        # Extract justification if available
        if "Justification:" in response:
            parts = response.split("Justification:", 1)
            if len(parts) > 1:
                justification = parts[1].strip()

        return selected_answer, confidence, justification


@jit(sample_input={"question": "What is 2+2?", "choices": {"A": "3", "B": "4"}})
class EnsembleJudgePipeline(Operator[MCQInput, EnsembleJudgeOutput]):
    """JIT-optimized pipeline combining ensemble and judge operators.

    This pipeline leverages JIT optimization to automatically parallelize independent
    operations in the execution graph. The structural JIT implementation analyzes the
    operator composition and identifies opportunities for concurrent execution.
    """

    # Complete specification with input/output models
    specification: ClassVar[Specification] = CoreSpecification(
        input_model=MCQInput, structured_output=EnsembleJudgeOutput
    )
    ensemble_operator: VariedEnsembleMCQOperator
    judge_operator: JudgeOperator

    def __init__(
        self,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        judge_model: str = "anthropic:claude-3-sonnet",
    ) -> None:
        """Initialize the pipeline with ensemble and judge operators.

        Args:
            model_configs: List of model configuration dictionaries for the ensemble
            judge_model: Name of the model to use for judging
        """
        self.ensemble_operator = VariedEnsembleMCQOperator(model_configs=model_configs)
        self.judge_operator = JudgeOperator(model_name=judge_model)

    def forward(self, *, inputs: MCQInput) -> EnsembleJudgeOutput:
        """Process a question through the ensemble and judge pipeline.

        The JIT optimizer automatically identifies that ensemble_operator can run
        parallel inference across multiple models, creating an optimized execution plan.

        Args:
            inputs: Question and choices

        Returns:
            The judge's selected answer and justification
        """
        # Get ensemble responses - JIT will automatically parallelize model calls internally
        # Pass MCQInput as an object, not a dict
        ensemble_outputs = self.ensemble_operator(inputs=inputs)

        # Prepare input for judge
        judge_input = EnsembleJudgeInput(
            question=inputs.question,
            choices=inputs.choices,
            candidate_responses=ensemble_outputs,
        )

        # Get judge decision
        return self.judge_operator(inputs=judge_input)


# Simplified pipeline creation function
def create_pipeline(
    model_configs: Optional[List[Dict[str, Any]]] = None,
    judge_model: str = "anthropic:claude-3-sonnet",
) -> EnsembleJudgePipeline:
    """Create a pipeline combining ensemble and judge operators.

    Args:
        model_configs: Model configurations for the ensemble
        judge_model: Model to use for judging

    Returns:
        Ensemble-judge pipeline instance
    """
    return EnsembleJudgePipeline(model_configs=model_configs, judge_model=judge_model)


class MMLUExperiment:
    """Experiment class for comparing baseline vs. ensemble-judge approaches on MMLU."""

    def __init__(
        self,
        subject: str = "high_school_mathematics",
        sample_size: int = 3,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        baseline_model: str = "anthropic:claude-3-haiku",
        judge_model: str = "anthropic:claude-3-sonnet",
        use_acceleration: bool = True,
    ) -> None:
        """Initialize the experiment.

        Args:
            subject: MMLU subject to evaluate
            sample_size: Number of samples to use
            model_configs: Model configurations for the ensemble
            baseline_model: Model for the baseline operator
            judge_model: Model for the judge operator
            use_acceleration: Whether to use XCS acceleration
        """
        self.subject = subject
        self.sample_size = sample_size

        # Initialize operators
        self.baseline_operator = BaselineMCQOperator(model_name=baseline_model)

        # Create a standard pipeline - use execution_options to control acceleration
        self.ensemble_judge_operator = EnsembleJudgePipeline(
            model_configs=model_configs, judge_model=judge_model
        )
        # Note: acceleration is controlled at execution time using execution_options

        # Load MMLU dataset
        self.mmlu_dataset = MMLUDataset(subject=subject)

    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results.

        Returns:
            Dictionary of experiment results
        """
        # Load data samples
        mmlu_data = self.mmlu_dataset.load(max_samples=self.sample_size)

        # Initialize results
        results = {
            "subject": self.subject,
            "sample_size": len(mmlu_data),
            "baseline": {
                "correct": 0,
                "total": len(mmlu_data),
                "time": 0,
                "detailed": [],
            },
            "ensemble_judge": {
                "correct": 0,
                "total": len(mmlu_data),
                "time": 0,
                "detailed": [],
            },
        }

        # Run baseline
        console.print(Panel(f"Running baseline on {self.subject}...", style="blue"))
        baseline_start = time.time()

        for item in mmlu_data:
            # Convert dict choices to key-value format
            choices = {k: v for k, v in item["choices"].items()}
            correct_answer = item["answer"]

            input_data = MCQInput(question=item["question"], choices=choices)
            output = self.baseline_operator(inputs=input_data)

            # Find which letter corresponds to the selected answer
            selected_letter = None
            for letter, text in choices.items():
                if text == output.answer:
                    selected_letter = letter
                    break

            # Check if the selected letter matches the correct answer letter
            is_correct = selected_letter == correct_answer

            results["baseline"]["detailed"].append(
                {
                    "question": item["question"],
                    "choices": choices,
                    "correct_answer": correct_answer,
                    "predicted": output.answer,
                    "is_correct": is_correct,
                    "reasoning": output.reasoning,
                }
            )

            if is_correct:
                results["baseline"]["correct"] += 1

        results["baseline"]["time"] = time.time() - baseline_start
        results["baseline"]["accuracy"] = (
            results["baseline"]["correct"] / results["baseline"]["total"]
        )

        # Run ensemble-judge
        console.print(
            Panel(f"Running ensemble-judge on {self.subject}...", style="green")
        )
        ensemble_start = time.time()

        for item in mmlu_data:
            # Convert dict choices to key-value format
            choices = {k: v for k, v in item["choices"].items()}
            correct_answer = item["answer"]

            input_data = MCQInput(question=item["question"], choices=choices)

            # Use execution_options context manager for parallel execution
            with execution_options(use_parallel=True):
                output = self.ensemble_judge_operator(inputs=input_data)

            # Find which letter corresponds to the selected answer
            selected_letter = None
            for letter, text in choices.items():
                if text == output.selected_answer:
                    selected_letter = letter
                    break

            # Check if the selected letter matches the correct answer letter
            is_correct = selected_letter == correct_answer

            results["ensemble_judge"]["detailed"].append(
                {
                    "question": item["question"],
                    "choices": choices,
                    "correct_answer": correct_answer,
                    "predicted": output.selected_answer,
                    "is_correct": is_correct,
                    "justification": output.justification,
                }
            )

            if is_correct:
                results["ensemble_judge"]["correct"] += 1

        results["ensemble_judge"]["time"] = time.time() - ensemble_start
        results["ensemble_judge"]["accuracy"] = (
            results["ensemble_judge"]["correct"] / results["ensemble_judge"]["total"]
        )

        # Record the number of models used in the ensemble for reference
        num_models = len(self.ensemble_judge_operator.ensemble_operator.lm_modules)
        results["num_models"] = num_models

        # We'll simply report the measured time for both approaches without
        # trying to calculate a theoretical speedup, as real-world API calls
        # have complex queueing, network effects, and provider-side processing
        # that make simple multipliers inaccurate

        return results


class ExperimentVisualizer:
    """Utilities for visualizing experiment results."""

    @staticmethod
    def print_summary(results: Dict[str, Any]) -> None:
        """Print a summary of the experiment results.

        Args:
            results: Dictionary of experiment results
        """
        console.print("\n")
        table = Table(title=f"MMLU Results: {results['subject']}")

        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", style="blue")
        table.add_column("Ensemble+Judge", style="green")

        table.add_row(
            "Accuracy",
            f"{results['baseline']['accuracy']:.2%}",
            f"{results['ensemble_judge']['accuracy']:.2%}",
        )

        table.add_row(
            "Execution Time",
            f"{results['baseline']['time']:.2f}s",
            f"{results['ensemble_judge']['time']:.2f}s",
        )

        if results.get("speedup"):
            relative_performance = (
                "↑"
                if results["ensemble_judge"]["accuracy"]
                > results["baseline"]["accuracy"]
                else "↓"
            )
            accuracy_diff = (
                abs(
                    results["ensemble_judge"]["accuracy"]
                    - results["baseline"]["accuracy"]
                )
                * 100
            )

            table.add_row(
                "Improvement",
                "Baseline",
                f"{relative_performance} {accuracy_diff:.2f}% accuracy",
            )

            # Show the number of models used, not a theoretical speedup
            table.add_row("Models Used", "1", f"{results.get('num_models', 'N/A')}")

        console.print(table)

    @staticmethod
    def plot_results(
        all_results: List[Dict[str, Any]], output_path: Optional[str] = None
    ) -> None:
        """Plot comparison of baseline vs ensemble-judge across subjects.

        Args:
            all_results: List of result dictionaries from experiments
            output_path: Path to save the plot image
        """
        subjects = [r["subject"].replace("_", " ").title() for r in all_results]
        baseline_accuracy = [r["baseline"]["accuracy"] for r in all_results]
        ensemble_accuracy = [r["ensemble_judge"]["accuracy"] for r in all_results]

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Accuracy comparison
        x = np.arange(len(subjects))
        width = 0.35

        ax1.bar(
            x - width / 2, baseline_accuracy, width, label="Baseline", color="royalblue"
        )
        ax1.bar(
            x + width / 2,
            ensemble_accuracy,
            width,
            label="Ensemble+Judge",
            color="forestgreen",
        )

        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy by Subject")
        ax1.set_xticks(x)
        ax1.set_xticklabels(subjects, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Time comparison
        baseline_time = [r["baseline"]["time"] for r in all_results]
        ensemble_time = [r["ensemble_judge"]["time"] for r in all_results]

        ax2.bar(
            x - width / 2, baseline_time, width, label="Baseline", color="royalblue"
        )
        ax2.bar(
            x + width / 2,
            ensemble_time,
            width,
            label="Ensemble+Judge",
            color="forestgreen",
        )

        ax2.set_ylabel("Execution Time (s)")
        ax2.set_title("Execution Time by Subject")
        ax2.set_xticks(x)
        ax2.set_xticklabels(subjects, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Speedup comparison
        speedups = [r.get("speedup", 1.0) for r in all_results]

        ax3.bar(x, speedups, width * 1.5, color="orange")
        ax3.set_ylabel("Speedup Factor")
        ax3.set_title("Acceleration Speedup vs Non-Accelerated")
        ax3.set_xticks(x)
        ax3.set_xticklabels(subjects, rotation=45, ha="right")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            console.print(f"Plot saved to {output_path}")

        plt.show()

    @staticmethod
    def plot_acceleration_comparison(
        benchmark_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> None:
        """Plot comparison of different acceleration strategies.

        Args:
            benchmark_results: Dictionary containing benchmark measurements
            output_path: Path to save the plot image
        """
        # Create figure with two subplots - execution time and relative speedup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Setup data
        strategies = ["Sequential", "Auto Parallel", "Explicit pmap"]
        colors = ["firebrick", "goldenrod", "forestgreen"]
        times = [
            benchmark_results["sequential_time"],
            benchmark_results["auto_parallel_time"],
            benchmark_results["pmap_time"],
        ]

        # Compute speedups relative to sequential execution
        speedups = [
            1.0,  # Sequential baseline
            benchmark_results["auto_parallel_speedup"],
            benchmark_results["pmap_speedup"],
        ]

        # Plot execution times
        bars1 = ax1.bar(strategies, times, color=colors)
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Execution Time by Strategy")
        ax1.grid(axis="y", alpha=0.3)

        # Add time values on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.4f}s",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot speedups
        bars2 = ax2.bar(strategies, speedups, color=colors)
        ax2.set_ylabel("Speedup Factor (vs Sequential)")
        ax2.set_title("Relative Speedup by Strategy")
        ax2.axhline(
            y=1.0, color="black", linestyle="--", alpha=0.5
        )  # Add reference line at 1.0
        ax2.grid(axis="y", alpha=0.3)

        # Add speedup values on top of bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{height:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            console.print(f"Acceleration comparison saved to {output_path}")

        plt.show()


def run_acceleration_benchmark(
    subject: str, sample_size: int, model_configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Benchmark parallel vs sequential execution performance.

    Compares three execution strategies:
    1. JIT with sequential execution (using execution_options to force sequential)
    2. JIT with automatic parallelism (default structural JIT behavior)
    3. JIT with pmap transform applied (explicit parallelism)

    Args:
        subject: MMLU subject to evaluate
        sample_size: Number of samples to use
        model_configs: Model configurations for the ensemble

    Returns:
        Dictionary of benchmark results
    """
    console.print(
        Panel(f"Benchmarking execution strategies on {subject}...", style="yellow")
    )

    mmlu_data = MMLUDataset(subject=subject).load(max_samples=sample_size)
    results = {}

    # Prepare test input
    if not mmlu_data:
        console.print("[red]No test data available![/red]")
        return {"subject": subject, "error": "No test data available"}

    sample_item = mmlu_data[0]
    test_input = MCQInput(
        question=sample_item["question"],
        choices={k: v for k, v in sample_item["choices"].items()},
    )

    # Create operators with different execution strategies
    # 1. Sequential JIT (force sequential execution)
    sequential_pipeline = EnsembleJudgePipeline(
        model_configs=model_configs[:2]  # Use just 2 models for benchmark
    )

    # 2. Automatic parallel JIT (default behavior)
    auto_parallel_pipeline = EnsembleJudgePipeline(
        model_configs=model_configs[:2]  # Use just 2 models for benchmark
    )

    # 3. For explicit parallelization, we'll just use execution_options with the same pipeline

    # Measure sequential execution time
    console.print("Running sequential execution benchmark...")
    sequential_start = time.perf_counter()  # Use perf_counter for high precision timing
    with execution_options(use_parallel=False):  # Force sequential execution
        sequential_result = sequential_pipeline(inputs=test_input)
    sequential_time = time.perf_counter() - sequential_start

    # Measure automatic parallel execution time
    console.print("Running automatic parallel execution benchmark...")
    auto_parallel_start = time.perf_counter()
    # No execution_options context - let JIT automatically parallelize
    auto_parallel_result = auto_parallel_pipeline(inputs=test_input)
    auto_parallel_time = time.perf_counter() - auto_parallel_start

    # Measure accelerated execution time (using execution_options consistently)
    console.print("Running explicitly accelerated execution benchmark...")
    pmap_start = time.perf_counter()
    with execution_options(use_parallel=True):  # Ensure parallel execution
        # We now use the same pipeline but with parallel execution enforced
        pmap_result = auto_parallel_pipeline(inputs=test_input)
    pmap_time = time.perf_counter() - pmap_start

    # Calculate time differences as percentages relative to sequential
    # This avoids implying a theoretical maximum speedup and focuses on actual measurements
    auto_parallel_diff = (
        ((sequential_time - auto_parallel_time) / sequential_time) * 100
        if sequential_time > 0
        else 0
    )
    pmap_diff = (
        ((sequential_time - pmap_time) / sequential_time) * 100
        if sequential_time > 0
        else 0
    )

    # Return benchmark results
    return {
        "subject": subject,
        "sequential_time": sequential_time,
        "auto_parallel_time": auto_parallel_time,
        "pmap_time": pmap_time,
        "auto_parallel_diff": auto_parallel_diff,
        "pmap_diff": pmap_diff,
        "sequential_answer": sequential_result.selected_answer,
        "auto_parallel_answer": auto_parallel_result.selected_answer,
        "pmap_answer": pmap_result.selected_answer,
    }


def main() -> None:
    """Main function to run the ensemble-judge MMLU evaluation example.

    Environment variables:
      SAMPLE_SIZE: Number of samples per subject (default: 2)
      MMLU_SUBJECTS: Comma-separated list of subjects (default: first 2)
    """
    # Get configuration from environment variables
    import os

    sample_size = int(os.environ.get("SAMPLE_SIZE", "2"))
    console.print(
        Panel.fit(
            "MMLU Evaluation: Baseline vs. Ensemble+Judge Pipeline with XCS Optimization",
            title="Ember Advanced Example",
            subtitle="Demonstrating LLM Ensemble Techniques and XCS Acceleration",
            style="bold green",
        )
    )

    # Check if API keys are set
    import os

    api_keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }

    if not any(api_keys.values()):
        # No API keys found, provide clear instructions
        console.print(
            Panel(
                "⚠️ No API keys found for language model providers.\n\n"
                "To run this example with real models, please set at least one of these environment variables:\n"
                "  - ANTHROPIC_API_KEY\n"
                "  - OPENAI_API_KEY\n"
                "  - GOOGLE_API_KEY\n\n"
                "Example:\n"
                "  export ANTHROPIC_API_KEY=your_api_key_here\n\n"
                "This example will showcase the code structure and architecture without making actual API calls.",
                title="API Keys Required",
                style="yellow",
            )
        )
        return

    # Define model configurations
    model_configs = [
        {"model_name": "anthropic:claude-3-haiku", "temperature": 0.0},
        {"model_name": "anthropic:claude-3-haiku", "temperature": 0.7},
        {"model_name": "openai:gpt-3.5-turbo", "temperature": 0.0},
    ]

    # Get subjects from environment with sensible default
    subjects_env = os.environ.get("MMLU_SUBJECTS", "")
    if subjects_env:
        subject_list = [s.strip() for s in subjects_env.split(",")]
        subjects_to_evaluate = [s for s in subject_list if s in MMLU_SUBJECTS]
        # If no valid subjects found, fall back to default
        if not subjects_to_evaluate:
            subjects_to_evaluate = MMLU_SUBJECTS[:2]
    else:
        subjects_to_evaluate = MMLU_SUBJECTS[:2]  # Default: first 2 subjects

    try:
        # Run experiments on selected subjects
        all_results = []

        for subject in subjects_to_evaluate:
            console.print(f"\n[bold]Running experiment on {subject}...[/bold]")

            # Use environment variable to control model count with a sensible default
            model_count = min(
                int(os.environ.get("MODEL_COUNT", "2")), len(model_configs)
            )

            experiment = MMLUExperiment(
                subject=subject,
                sample_size=sample_size,
                model_configs=model_configs[:model_count],
                use_acceleration=True,
            )

            results = experiment.run()
            all_results.append(results)

            # Print summary for this subject
            ExperimentVisualizer.print_summary(results)

        # Run acceleration benchmarks to compare different strategies
        console.print("\n[bold]Benchmarking XCS acceleration strategies:[/bold]")
        # For benchmark, use same model count as experiment
        benchmark_results = run_acceleration_benchmark(
            subject=subjects_to_evaluate[0],  # Use first subject
            sample_size=sample_size,
            model_configs=model_configs[:model_count],
        )

        # Print comparison
        acc_table = Table(title="XCS Acceleration Strategy Comparison")
        acc_table.add_column("Metric", style="cyan")
        acc_table.add_column("Sequential", style="red")
        acc_table.add_column("Auto Parallel", style="yellow")
        acc_table.add_column("Explicit pmap", style="green")

        acc_table.add_row(
            "Execution Time",
            f"{benchmark_results['sequential_time']:.4f}s",
            f"{benchmark_results['auto_parallel_time']:.4f}s",
            f"{benchmark_results['pmap_time']:.4f}s",
        )

        acc_table.add_row(
            "Speedup",
            "Baseline",
            f"{benchmark_results['auto_parallel_speedup']:.2f}x",
            f"{benchmark_results['pmap_speedup']:.2f}x",
        )

        # Visualize acceleration benchmark results using our specialized visualization
        console.print("\n[bold]Acceleration Benchmark Visualization:[/bold]")
        try:
            ExperimentVisualizer.plot_acceleration_comparison(
                benchmark_results=benchmark_results,
                output_path="acceleration_strategies.png",
            )
            console.print(
                f"[green]Acceleration visualization saved to acceleration_strategies.png[/green]"
            )
        except Exception as e:
            console.print(f"[yellow]Visualization error: {e}[/yellow]")

        console.print(acc_table)

        # Skip visualizing results if matplotlib is not available
        try:
            # Visualize results if we have enough data
            if len(all_results) > 1:
                console.print("\n[bold]Generating visualization...[/bold]")
                ExperimentVisualizer.plot_results(all_results)
        except Exception as e:
            console.print(f"[yellow]Skipping visualization: {e}[/yellow]")

        console.print("\n[bold green]Example completed successfully![/bold green]")

    except Exception as e:
        console.print(
            Panel(
                f"Error running example: {str(e)}\n\n"
                "This example requires properly configured API keys and model availability.\n"
                "Please check your API keys and available models.",
                title="Execution Error",
                style="red",
            )
        )


if __name__ == "__main__":
    main()
