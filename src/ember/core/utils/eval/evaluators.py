from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar, Optional, List

T_out = TypeVar("T_out")
T_truth = TypeVar("T_truth")


@dataclass
class EvaluationResult:
    """Encapsulates the result of evaluating a system output against a reference.

    Attributes:
        is_correct (bool): Whether the system output meets the expected criteria.
        score (float): A numeric score reflecting accuracy or quality.
        metadata (Optional[Dict[str, Any]]): Optional additional details about the evaluation.
    """

    is_correct: bool
    score: float
    metadata: Optional[Dict[str, Any]] = None


class IEvaluator(ABC, Generic[T_out]):
    """Interface for evaluating a system output against a reference answer.

    Attributes:
        T_out: The type of the system output.
    """

    @abstractmethod
    def evaluate(
        self, system_output: T_out, correct_answer: Any, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluates the system output compared to the expected correct answer.

        Args:
            system_output (T_out): The raw output produced by the system.
            correct_answer (Any): The expected answer.
            **kwargs (Any): Additional parameters specific to the evaluator.

        Returns:
            EvaluationResult: The result of the evaluation.
        """
        raise NotImplementedError


@dataclass
class ExtractionResult:
    """Holds the result of extracting or parsing system output.

    Attributes:
        extracted_value (Any): The value parsed from the system output.
        metadata (Optional[Dict[str, Any]]): Optional additional extraction details.
    """

    extracted_value: Any
    metadata: Optional[Dict[str, Any]] = None


class IOutputExtractor(ABC, Generic[T_out, T_truth]):
    """Interface for extracting or converting raw system output into a refined form."""

    @abstractmethod
    def extract(self, system_output: T_out, **kwargs: Any) -> T_truth:
        """Extracts and converts the system output into a form suitable for evaluation.

        Args:
            system_output (T_out): The raw system output.
            **kwargs (Any): Additional parameters for extraction.

        Returns:
            T_truth: The extracted or converted output.
        """
        raise NotImplementedError


OutType = TypeVar("OutType")
ExtractedType = TypeVar("ExtractedType")


class ComposedEvaluator(Generic[OutType, ExtractedType], IEvaluator[OutType]):
    """Combines an output extractor with an evaluator for the extracted data.

    The extractor converts T_out to a more manageable form which the evaluator then processes.
    """

    def __init__(
        self,
        extractor: IOutputExtractor[OutType, ExtractedType],
        base_evaluator: IEvaluator[ExtractedType],
    ) -> None:
        """Initializes the composed evaluator.

        Args:
            extractor (IOutputExtractor[OutType, ExtractedType]): The component that extracts required data.
            base_evaluator (IEvaluator[ExtractedType]): The evaluator that processes the extracted data.
        """
        self.extractor: IOutputExtractor[OutType, ExtractedType] = extractor
        self.base_evaluator: IEvaluator[ExtractedType] = base_evaluator

    def evaluate(
        self, system_output: OutType, correct_answer: Any, **kwargs: Any
    ) -> EvaluationResult:
        """Extracts information from the system output and then evaluates it.

        Args:
            system_output (OutType): The raw system output.
            correct_answer (Any): The expected answer.
            **kwargs (Any): Additional parameters for the extraction and evaluation.

        Returns:
            EvaluationResult: The result after both extraction and evaluation.
        """
        extracted_value: ExtractedType = self.extractor.extract(system_output, **kwargs)
        return self.base_evaluator.evaluate(extracted_value, correct_answer, **kwargs)


# Concrete Evaluators


class ExactMatchEvaluator(IEvaluator[str]):
    """Evaluator that checks if two strings match exactly (ignoring whitespace and case)."""

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Compares the system output against the correct answer after normalizing both strings.

        Args:
            system_output (str): The output string produced by the system.
            correct_answer (str): The expected string.
            **kwargs (Any): Additional parameters (currently unused).

        Returns:
            EvaluationResult: Full score if matched, zero score otherwise.
        """
        normalized_output: str = system_output.strip().lower()
        normalized_answer: str = correct_answer.strip().lower()
        is_correct: bool = normalized_output == normalized_answer
        score: float = 1.0 if is_correct else 0.0
        return EvaluationResult(is_correct=is_correct, score=score)


class NumericToleranceEvaluator(IEvaluator[float]):
    """Evaluator that checks if a numeric output is within a specified tolerance of the expected value."""

    def __init__(self, tolerance: float = 0.01) -> None:
        """Initializes the numeric evaluator with a tolerance threshold.

        Args:
            tolerance (float): The acceptable difference between output and expected value.
        """
        self.tolerance: float = tolerance

    def evaluate(
        self, system_output: float, correct_answer: float, **kwargs: Any
    ) -> EvaluationResult:
        """Determines if the numeric output falls within the acceptable tolerance of the correct answer.

        Args:
            system_output (float): The numeric output from the system.
            correct_answer (float): The expected numeric value.
            **kwargs (Any): Additional parameters (currently unused).

        Returns:
            EvaluationResult: The result including the difference in metadata.
        """
        difference: float = abs(system_output - correct_answer)
        is_correct: bool = difference <= self.tolerance
        base: float = abs(correct_answer) if correct_answer != 0 else 1.0
        score: float = max(0.0, 1.0 - difference / base)
        return EvaluationResult(
            is_correct=is_correct, score=score, metadata={"diff": difference}
        )


class CodeExecutionEvaluator(IEvaluator[str]):
    """Evaluator that executes Python code and compares its standard output to an expected result."""

    def __init__(self, timeout: float = 5.0) -> None:
        """Sets up the code execution evaluator with a timeout.

        Args:
            timeout (float): Maximum allowed time (in seconds) for code execution.
        """
        self.timeout: float = timeout

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Executes the provided Python code and compares its output to the expected string.

        Args:
            system_output (str): The Python code to be executed.
            correct_answer (str): The expected output from running the code.
            **kwargs (Any): Additional parameters (currently unused).

        Returns:
            EvaluationResult: The result including stdout, stderr, and exit code details.
        """
        try:
            process_result: subprocess.CompletedProcess = subprocess.run(
                args=["python", "-c", system_output],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout_str: str = process_result.stdout.strip()
            expected_str: str = correct_answer.strip()
            is_correct: bool = stdout_str == expected_str
            return EvaluationResult(
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                metadata={
                    "stdout": process_result.stdout,
                    "stderr": process_result.stderr,
                    "exit_code": process_result.returncode,
                },
            )
        except Exception as error:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": str(error)},
            )


class RegexExtractor(IOutputExtractor[str, str]):
    """Extractor that uses a regular expression to select a substring.

    It returns the first captured group, or an empty string if no match is found.
    """

    def __init__(self, pattern: str) -> None:
        """Compiles the regex pattern.

        Args:
            pattern (str): The regex pattern that should include at least one capturing group.
        """
        self.compiled_pattern: re.Pattern = re.compile(pattern)

    def extract(self, system_output: str, **kwargs: Any) -> str:
        """Extracts the first matching subgroup from the given system output.

        Args:
            system_output (str): The string to search.
            **kwargs (Any): Additional parameters (currently unused).

        Returns:
            str: The extracted substring if a match is found, otherwise an empty string.
        """
        match: Optional[re.Match] = self.compiled_pattern.search(system_output)
        if match is None:
            return ""
        return match.group(1)


# Composite Evaluator Example


class PartialRegexEvaluator(ComposedEvaluator[str, str]):
    """Combines a RegexExtractor with an ExactMatchEvaluator.

    First, extracts a substring via regex, then verifies an exact match against the expected value.
    """

    def __init__(self, pattern: str) -> None:
        """Initializes the composite evaluator with the provided regex pattern.

        Args:
            pattern (str): The regex pattern used to extract the relevant substring.
        """
        extractor: RegexExtractor = RegexExtractor(pattern)
        evaluator: ExactMatchEvaluator = ExactMatchEvaluator()
        super().__init__(extractor=extractor, base_evaluator=evaluator)


# Evaluator Registry and Batch Evaluation


class EvaluatorRegistry:
    """Registry to store and retrieve evaluators by a unique name."""

    def __init__(self) -> None:
        """Initializes the registry with an empty dictionary."""
        self._evaluators: Dict[str, IEvaluator[Any]] = {}

    def register(self, name: str, evaluator: IEvaluator[Any]) -> None:
        """Registers an evaluator under the given name.

        Args:
            name (str): The unique name to associate with the evaluator.
            evaluator (IEvaluator[Any]): The evaluator instance to register.
        """
        self._evaluators[name] = evaluator

    def get(self, name: str) -> IEvaluator[Any]:
        """Retrieves an evaluator by its name.

        Args:
            name (str): The name of the evaluator.

        Returns:
            IEvaluator[Any]: The evaluator associated with the provided name.

        Raises:
            KeyError: If no evaluator exists with the given name.
        """
        if name not in self._evaluators:
            raise KeyError(f"No evaluator found with name: {name}")
        return self._evaluators[name]


def evaluate_batch(
    evaluator: IEvaluator[Any],
    system_outputs: List[Any],
    correct_answers: List[Any],
    **kwargs: Any,
) -> List[EvaluationResult]:
    """Evaluates a batch of system outputs against their corresponding correct answers.

    Args:
        evaluator (IEvaluator[Any]): The evaluator to apply.
        system_outputs (List[Any]): A list of system outputs.
        correct_answers (List[Any]): A list of expected correct answers.
        **kwargs (Any): Additional parameters for the evaluation.

    Returns:
        List[EvaluationResult]: A list containing the evaluation results.
    """
    results: List[EvaluationResult] = []
    for system_output, correct_answer in zip(system_outputs, correct_answers):
        result: EvaluationResult = evaluator.evaluate(
            system_output, correct_answer, **kwargs
        )
        results.append(result)
    return results


if __name__ == "__main__":
    # Example 1: Direct final-output comparison (exact match)
    exact_evaluator: ExactMatchEvaluator = ExactMatchEvaluator()
    result_exact: EvaluationResult = exact_evaluator.evaluate(
        "Hello World", "hello  world"
    )
    print("ExactMatch result:", result_exact)

    # Example 2: Numeric tolerance evaluation
    numeric_evaluator: NumericToleranceEvaluator = NumericToleranceEvaluator(
        tolerance=0.05
    )
    result_numeric: EvaluationResult = numeric_evaluator.evaluate(3.14159, 3.14)
    print("NumericTolerance result:", result_numeric)

    # Example 3: Composite evaluator with regex extraction and exact matching.
    regex_pattern: str = r"answer\s+is\s+(\w+)"
    partial_regex_evaluator: PartialRegexEvaluator = PartialRegexEvaluator(
        pattern=regex_pattern
    )
    result_regex: EvaluationResult = partial_regex_evaluator.evaluate(
        "The answer is PARIS", "PARIS"
    )
    print("PartialRegexEvaluator result:", result_regex)

    # Example 4: Code execution evaluator.
    code_evaluator: CodeExecutionEvaluator = CodeExecutionEvaluator()
    code_string: str = "print('Hello')"
    result_code: EvaluationResult = code_evaluator.evaluate(code_string, "Hello")
    print("CodeExecutionEvaluator result:", result_code)

    # Example 5: Batch evaluation using the evaluator registry.
    evaluator_registry: EvaluatorRegistry = EvaluatorRegistry()
    evaluator_registry.register("exact", exact_evaluator)
    evaluator_registry.register("regex", partial_regex_evaluator)

    system_outputs: List[str] = ["I think the answer is PARIS", "The answer is LONDON"]
    corrects: List[str] = ["PARIS", "LONDON"]
    batch_results: List[EvaluationResult] = evaluate_batch(
        evaluator_registry.get("regex"), system_outputs, corrects
    )
    print("Batch results:", batch_results)
