from __future__ import annotations

import subprocess
from typing import Any, Callable, Generic, Optional, TypeVar

from .base_evaluator import EvaluationResult, IEvaluator
from .extractors import RegexExtractor

T_out = TypeVar("T_out")
T_truth = TypeVar("T_truth")


class ComposedEvaluator(IEvaluator[T_out, T_truth], Generic[T_out, T_truth]):
    """Combines an output extractor with an evaluator for the extracted data.

    This evaluator first transforms the system output using the provided extractor,
    then evaluates the extracted value using the specified base evaluator.

    Args:
        extractor: An object with an `extract` method to process the system output.
        base_evaluator (IEvaluator): An evaluator that processes the extracted output.

    Returns:
        EvaluationResult: The result of the evaluation.
    """

    def __init__(
        self,
        extractor: Any,  # Expecting an extractor with an `extract` method.
        base_evaluator: IEvaluator[Any, Any],
    ) -> None:
        self.extractor = extractor
        self.base_evaluator = base_evaluator

    def evaluate(
        self, system_output: T_out, correct_answer: Any, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluates the provided system output against the correct answer.

        Args:
            system_output (T_out): The raw output generated by the system.
            correct_answer (Any): The expected correct answer.
            **kwargs: Additional keyword arguments for extraction or evaluation.

        Returns:
            EvaluationResult: The result of evaluating the extracted value.
        """
        extracted_value = self.extractor.extract(system_output, **kwargs)
        return self.base_evaluator.evaluate(extracted_value, correct_answer, **kwargs)


# Basic Evaluators


class ExactMatchEvaluator(IEvaluator[str, str]):
    """Evaluator to check for an exact match between two strings,
    ignoring differences in whitespace and case.

    Example:
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate("Hello World", "hello   world")

    Args:
        compare_fn (Optional[Callable[[str, str], bool]]): Optional custom comparison function.
            If not provided, strings are normalized (whitespace removed, lowercase) before comparison.

    Returns:
        EvaluationResult: The result containing a correctness flag and a score.
    """

    def __init__(self, compare_fn: Optional[Callable[[str, str], bool]] = None) -> None:
        self.compare_fn = compare_fn or self._default_compare

    def _default_compare(self, str1: str, str2: str) -> bool:
        """Default string comparison function that ignores case and whitespace.

        Args:
            str1 (str): First string to compare
            str2 (str): Second string to compare

        Returns:
            bool: True if strings match after normalization
        """
        return str1.strip().lower() == str2.strip().lower()

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluates whether a system output exactly matches the correct answer.

        Args:
            system_output (str): The system-generated string.
            correct_answer (str): The expected answer string.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            EvaluationResult: An object with `is_correct` set to True if the normalized strings match,
                              along with a corresponding score.
        """
        is_correct = self.compare_fn(system_output, correct_answer)
        score = 1.0 if is_correct else 0.0
        return EvaluationResult(is_correct=is_correct, score=score)


class NumericToleranceEvaluator(IEvaluator[float, float]):
    """Evaluator to check if a numeric output is within a specified tolerance of the expected value.

    Example:
        evaluator = NumericToleranceEvaluator(tolerance=0.05)
        result = evaluator.evaluate(3.14159, 3.14)

    Args:
        tolerance (float): The maximum allowed difference between the output and the correct value.
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    def evaluate(
        self, system_output: float, correct_answer: float, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluates the numeric system output against the correct value within a specified tolerance.

        Args:
            system_output (float): The numeric output from the system.
            correct_answer (float): The expected numeric answer.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            EvaluationResult: The result including a correctness flag, score, and metadata about the difference.
        """
        difference = abs(system_output - correct_answer)
        # Round to handle floating point precision issues
        rounded_diff = round(difference, 8)
        is_correct = rounded_diff <= self.tolerance
        base = abs(correct_answer) if correct_answer != 0 else 1.0
        score = max(0.0, 1.0 - rounded_diff / base)
        return EvaluationResult(
            is_correct=is_correct, score=score, metadata={"diff": rounded_diff}
        )


class CodeExecutionEvaluator(IEvaluator[str, str]):
    """Evaluator that executes Python code and compares its standard output to an expected result.

    **WARNING**: Executing arbitrary code is dangerous.
    Only use this evaluator with fully trusted code strings.

    Args:
        timeout (float): Maximum duration (in seconds) to allow code execution.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Executes the provided Python code and compares its standard output to the expected result.

        Args:
            system_output (str): A Python code string to be executed.
            correct_answer (str): The expected output from the code execution.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            EvaluationResult: The result of execution, including stdout, stderr, and exit code in metadata.
        """
        try:
            process_result: subprocess.CompletedProcess = subprocess.run(
                args=["python", "-c", system_output],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout_str = process_result.stdout.strip()
            expected_str = correct_answer.strip()
            is_correct = stdout_str == expected_str
            return EvaluationResult(
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                metadata={
                    "stdout": process_result.stdout,
                    "stderr": process_result.stderr,
                    "exit_code": process_result.returncode,
                },
            )
        except subprocess.TimeoutExpired as timeout_error:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": f"TimeoutExpired: {str(timeout_error)}"},
            )
        except Exception as error:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": f"{type(error).__name__}: {str(error)}"},
            )


# Composite Evaluator Example


class MultipleChoiceEvaluator(IEvaluator[str, str]):
    """Evaluator to check if a system output contains the correct multiple-choice answer.

    Searches for any occurrence of the correct choice (e.g., "A", "B", "C", "D") in the
    system output text, allowing for different formats like "The answer is A" or
    just "A".

    Example:
        evaluator = MultipleChoiceEvaluator()
        result = evaluator.evaluate("I think the answer is C because...", "C")
    """

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluates whether the system output contains the correct multiple-choice answer.

        Args:
            system_output (str): The system-generated text that should contain an answer.
            correct_answer (str): The expected answer letter or identifier (e.g., "A").
            **kwargs: Additional keyword arguments (unused).

        Returns:
            EvaluationResult: An object with `is_correct` set to True if the correct
                              answer is found in the system output.
        """
        # Clean and normalize the system output and correct answer
        system_output = system_output.strip().upper()
        correct_answer = correct_answer.strip().upper()

        # Look for answer patterns: standalone letter, or letter with context
        patterns = [
            rf"\b{correct_answer}\b",  # Exact letter match
            rf"(?:ANSWER|OPTION|CHOICE)\s+IS\s+{correct_answer}",  # "Answer is X"
            rf"{correct_answer}\)",  # "A)" format
            rf"\({correct_answer}\)",  # "(A)" format
        ]

        # Check if any pattern matches
        import re

        for pattern in patterns:
            if re.search(pattern, system_output):
                return EvaluationResult(is_correct=True, score=1.0)

        # No match found
        return EvaluationResult(is_correct=False, score=0.0)


class PartialRegexEvaluator(ComposedEvaluator[str, str]):
    """Evaluator that uses a regex extractor followed by an exact match evaluation.

    First, it extracts a substring using a regular expression, then checks if the extracted
    value matches the expected answer exactly.

    Args:
        pattern (str): The regular expression pattern used for extraction.
    """

    def __init__(self, pattern: str) -> None:
        extractor = RegexExtractor(pattern)
        evaluator = ExactMatchEvaluator()
        super().__init__(extractor=extractor, base_evaluator=evaluator)


if __name__ == "__main__":
    # Example 1: Direct final-output comparison (exact match)
    exact_evaluator = ExactMatchEvaluator()
    result_exact = exact_evaluator.evaluate("Hello World", "hello  world")
    print("ExactMatch result:", result_exact)

    # Example 2: Numeric tolerance evaluation
    numeric_evaluator = NumericToleranceEvaluator(tolerance=0.05)
    result_numeric = numeric_evaluator.evaluate(3.14159, 3.14)
    print("NumericTolerance result:", result_numeric)

    # Example 3: Composite evaluator with regex extraction and exact matching.
    regex_pattern = r"answer\s+is\s+(\w+)"
    partial_regex_evaluator = PartialRegexEvaluator(pattern=regex_pattern)
    result_regex = partial_regex_evaluator.evaluate("The answer is PARIS", "PARIS")
    print("PartialRegexEvaluator result:", result_regex)

    # Example 4: Code execution evaluator.
    code_evaluator = CodeExecutionEvaluator()
    code_string = "print('Hello')"
    result_code = code_evaluator.evaluate(code_string, "Hello")
    print("CodeExecutionEvaluator result:", result_code)
