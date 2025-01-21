# evaluators.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar, Optional
import subprocess
import re


##########################################################
# 1) Core Data Structures & Interfaces
##########################################################

T_out = TypeVar("T_out")     # The type of the system's output
T_truth = TypeVar("T_truth") # The type of the correct answer


@dataclass
class EvaluationResult:
    """
    Encapsulates the result of evaluating a system output against a reference.
    """
    is_correct: bool
    score: float
    metadata: Dict[str, Any] = None


class IEvaluator(ABC, Generic[T_out]):
    """
    A generic interface for evaluating a system output (T_out) 
    against a 'correct_answer', returning an EvaluationResult.

    By design:
      - T_out is typically a str or dict, but can be anything 
      - 'correct_answer' can also be arbitrary, typed or untyped
    """

    @abstractmethod
    def evaluate(
        self,
        system_output: T_out,
        correct_answer: Any,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate 'system_output' vs. 'correct_answer'.
        Additional evaluator-specific parameters can be passed via kwargs.
        """
        pass


@dataclass
class ExtractionResult:
    """
    For advanced usage: if you do extraction steps, you might store 
    intermediate parse results or relevant info here.
    """
    extracted_value: Any
    metadata: Dict[str, Any] = None


class IOutputExtractor(ABC, Generic[T_out, T_truth]):
    """
    Optional interface for extracting or parsing the raw system output (T_out)
    into a typed or simplified form (T_truth) suitable for evaluation.
    """

    @abstractmethod
    def extract(self, system_output: T_out, **kwargs) -> T_truth:
        """
        Convert or parse the system_output into the type used by the evaluator.
        """
        pass


##########################################################
# 2) Composed Evaluator (Extractor + Evaluator)
##########################################################

OutType = TypeVar("OutType") 
ExtractedType = TypeVar("ExtractedType")

class ComposedEvaluator(Generic[OutType, ExtractedType], IEvaluator[OutType]):
    """
    Merges an IOutputExtractor with an IEvaluator for the extracted type.

    Flow:
      - Extractor: T_out => T_extracted
      - Evaluator: (T_extracted, correct_answer) => EvaluationResult
    """

    def __init__(
        self,
        extractor: IOutputExtractor[OutType, ExtractedType],
        base_evaluator: IEvaluator[ExtractedType]
    ):
        self.extractor = extractor
        self.base_evaluator = base_evaluator

    def evaluate(
        self,
        system_output: OutType,
        correct_answer: Any,
        **kwargs
    ) -> EvaluationResult:
        extracted_value = self.extractor.extract(system_output, **kwargs)
        return self.base_evaluator.evaluate(extracted_value, correct_answer, **kwargs)


##########################################################
# 3) Sample Concrete Evaluators
##########################################################

class ExactMatchEvaluator(IEvaluator[str]):
    """
    Evaluates correctness by checking if system_output (string)
    matches correct_answer (string), ignoring case + whitespace.
    """
    def evaluate(
        self,
        system_output: str,
        correct_answer: str,
        **kwargs
    ) -> EvaluationResult:
        out_clean = system_output.strip().lower()
        ans_clean = correct_answer.strip().lower()
        is_correct = (out_clean == ans_clean)
        return EvaluationResult(is_correct=is_correct, score=1.0 if is_correct else 0.0)


class NumericToleranceEvaluator(IEvaluator[float]):
    """
    Checks if system_output (float) is within tolerance of correct_answer (float).
    """
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def evaluate(
        self,
        system_output: float,
        correct_answer: float,
        **kwargs
    ) -> EvaluationResult:
        diff = abs(system_output - correct_answer)
        is_correct = diff <= self.tolerance
        # Score decreases as diff grows, but is never negative
        score = max(0, 1 - diff / (abs(correct_answer) if correct_answer != 0 else 1.0))
        return EvaluationResult(is_correct=is_correct, score=score, metadata={"diff": diff})


class CodeExecutionEvaluator(IEvaluator[str]):
    """
    Example evaluator that tries to run system_output as Python code 
    and compare stdout to a 'correct_answer' string.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def evaluate(
        self,
        system_output: str,
        correct_answer: str,
        **kwargs
    ) -> EvaluationResult:
        try:
            proc = subprocess.run(
                ["python", "-c", system_output],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            stdout_str = proc.stdout.strip()
            expected_str = correct_answer.strip()
            is_correct = (stdout_str == expected_str)
            return EvaluationResult(
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                metadata={
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "exit_code": proc.returncode
                }
            )
        except Exception as e:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": str(e)}
            )


##########################################################
# 4) Sample Extractor(s)
##########################################################

class RegexExtractor(IOutputExtractor[str, str]):
    """
    Extracts a substring from system_output (str) via a regex pattern,
    returning the first capturing group or an empty string if none found.
    """

    def __init__(self, pattern: str):
        self.compiled = re.compile(pattern)

    def extract(self, system_output: str, **kwargs) -> str:
        match = self.compiled.search(system_output)
        if not match:
            return ""
        return match.group(1)


##########################################################
# 5) Example: Combining Extraction + Evaluation
##########################################################

class PartialRegexEvaluator(ComposedEvaluator[str, str]):
    """
    Demonstrates combining a RegexExtractor with an ExactMatchEvaluator.
    This will parse a substring from the system output, then do an exact match.
    """

    def __init__(self, pattern: str):
        extractor = RegexExtractor(pattern)
        evaluator = ExactMatchEvaluator()
        super().__init__(extractor=extractor, base_evaluator=evaluator)


##########################################################
# 6) (Optional) Evaluator Registry or Batch Execution
##########################################################

class EvaluatorRegistry:
    """
    Optional: store named evaluators for easy reuse.
    """
    def __init__(self):
        self._evaluators: Dict[str, IEvaluator[Any]] = {}

    def register(self, name: str, evaluator: IEvaluator[Any]) -> None:
        self._evaluators[name] = evaluator

    def get(self, name: str) -> IEvaluator[Any]:
        if name not in self._evaluators:
            raise KeyError(f"No evaluator found with name: {name}")
        return self._evaluators[name]


def evaluate_batch(
    evaluator: IEvaluator[Any],
    system_outputs: list[Any],
    correct_answers: list[Any],
    **kwargs
) -> list[EvaluationResult]:
    """
    Helper to run the same evaluator over a list of (system_output, correct_answer) pairs.
    """
    results = []
    for out, ans in zip(system_outputs, correct_answers):
        res = evaluator.evaluate(out, ans, **kwargs)
        results.append(res)
    return results


##########################################################
# 7) Usage Example
##########################################################
if __name__ == "__main__":
    # Example 1: Direct final-output comparison (exact match)
    eval_exact = ExactMatchEvaluator()
    res1 = eval_exact.evaluate("Hello World", "hello  world")
    print("ExactMatch result:", res1)

    # Example 2: Numeric tolerance
    eval_num = NumericToleranceEvaluator(tolerance=0.05)
    res2 = eval_num.evaluate(3.14159, 3.14)
    print("NumericTolerance result:", res2)

    # Example 3: Combine a regex extraction + exact match
    # e.g. parse 'The answer is XXX' 
    pattern = r"answer\s+is\s+(\w+)"
    eval_regex = PartialRegexEvaluator(pattern=pattern)
    res3 = eval_regex.evaluate("The answer is PARIS", "PARIS")
    print("PartialRegexEvaluator result:", res3)

    # Example 4: Code execution evaluator
    code_eval = CodeExecutionEvaluator()
    # Suppose correct_answer is the expected stdout
    code_str = "print('Hello')"
    res4 = code_eval.evaluate(code_str, "Hello")
    print("CodeExecutionEvaluator result:", res4)

    # Example 5: Batch evaluation
    registry = EvaluatorRegistry()
    registry.register("exact", eval_exact)
    registry.register("regex", eval_regex)

    system_outs = ["I think the answer is PARIS", "The answer is LONDON"]
    corrects = ["PARIS", "LONDON"]
    # use the 'regex' evaluator from registry
    batch_res = evaluate_batch(registry.get("regex"), system_outs, corrects)
    print("Batch results:", batch_res)