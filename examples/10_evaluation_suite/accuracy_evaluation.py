"""Accuracy evaluation patterns for LLM outputs.

This example demonstrates:
- Basic evaluator interfaces
- Exact match evaluation
- Numeric tolerance evaluation
- Multiple choice evaluation
- Custom evaluator implementation

Run with:
    python examples/10_evaluation_suite/accuracy_evaluation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ember.utils.eval import (
    EvaluationResult,
    ExactMatchEvaluator,
    IEvaluator,
    NumericToleranceEvaluator,
)


# =============================================================================
# Part 1: Basic Evaluation Interface
# =============================================================================

def demonstrate_evaluation_result() -> None:
    """Show the EvaluationResult structure."""
    print("Part 1: Evaluation Result Structure")
    print("-" * 50)

    # Simple correct result
    correct_result = EvaluationResult(
        is_correct=True,
        score=1.0,
        metadata={"reason": "exact match"},
    )
    print(f"Correct result: is_correct={correct_result.is_correct}, score={correct_result.score}")

    # Partial credit result
    partial_result = EvaluationResult(
        is_correct=False,
        score=0.75,
        metadata={"reason": "partial match", "matched_tokens": 3, "total_tokens": 4},
    )
    print(f"Partial result: is_correct={partial_result.is_correct}, score={partial_result.score}")
    print(f"  Metadata: {partial_result.metadata}")
    print()


# =============================================================================
# Part 2: Exact Match Evaluation
# =============================================================================

def demonstrate_exact_match() -> None:
    """Show exact match evaluation."""
    print("Part 2: Exact Match Evaluation")
    print("-" * 50)

    evaluator = ExactMatchEvaluator()

    test_cases = [
        ("Paris", "Paris", "Exact match"),
        ("paris", "Paris", "Case mismatch"),
        ("Paris ", "Paris", "Trailing space"),
        ("The answer is Paris", "Paris", "Extra text"),
    ]

    for system_output, expected, description in test_cases:
        result = evaluator.evaluate(system_output, expected)
        status = "PASS" if result.is_correct else "FAIL"
        print(f"  [{status}] {description}")
        print(f"    Output: '{system_output}' vs Expected: '{expected}'")
        print(f"    Score: {result.score}")
    print()


# =============================================================================
# Part 3: Numeric Tolerance Evaluation
# =============================================================================

def demonstrate_numeric_tolerance() -> None:
    """Show numeric tolerance evaluation."""
    print("Part 3: Numeric Tolerance Evaluation")
    print("-" * 50)

    # Absolute tolerance
    abs_evaluator = NumericToleranceEvaluator(tolerance=0.01)

    # Relative tolerance (1%)
    rel_evaluator = NumericToleranceEvaluator(tolerance=0.01, relative=True)

    test_cases = [
        (3.14159, 3.14, "Pi approximation"),
        (100.0, 100.05, "Small absolute difference"),
        (1000.0, 1010.0, "1% relative difference"),
    ]

    print("Absolute tolerance (0.01):")
    for output, expected, description in test_cases:
        result = abs_evaluator.evaluate(output, expected)
        status = "PASS" if result.is_correct else "FAIL"
        print(f"  [{status}] {description}: {output} vs {expected}")

    print("\nRelative tolerance (1%):")
    for output, expected, description in test_cases:
        result = rel_evaluator.evaluate(output, expected)
        status = "PASS" if result.is_correct else "FAIL"
        print(f"  [{status}] {description}: {output} vs {expected}")
    print()


# =============================================================================
# Part 4: Custom Evaluator
# =============================================================================

@dataclass
class ContainsKeywordEvaluator(IEvaluator[str, str]):
    """Evaluator that checks if output contains expected keyword."""

    case_sensitive: bool = False

    def evaluate(
        self,
        system_output: str,
        correct_answer: str,
        **kwargs: object,
    ) -> EvaluationResult:
        """Check if system output contains the expected keyword."""
        if self.case_sensitive:
            contains = correct_answer in system_output
        else:
            contains = correct_answer.lower() in system_output.lower()

        return EvaluationResult(
            is_correct=contains,
            score=1.0 if contains else 0.0,
            metadata={"keyword": correct_answer, "found": contains},
        )


def demonstrate_custom_evaluator() -> None:
    """Show custom evaluator implementation."""
    print("Part 4: Custom Evaluator")
    print("-" * 50)

    evaluator = ContainsKeywordEvaluator(case_sensitive=False)

    test_cases = [
        ("The capital of France is Paris.", "Paris"),
        ("I think it might be London.", "Paris"),
        ("PARIS is the answer", "Paris"),
    ]

    for output, keyword in test_cases:
        result = evaluator.evaluate(output, keyword)
        status = "PASS" if result.is_correct else "FAIL"
        print(f"  [{status}] Looking for '{keyword}'")
        print(f"    In: '{output}'")
    print()


# =============================================================================
# Part 5: Weighted Scoring
# =============================================================================

@dataclass
class WeightedEvaluator(IEvaluator[Dict[str, Any], Dict[str, Any]]):
    """Evaluator with weighted criteria."""

    weights: Dict[str, float]

    def evaluate(
        self,
        system_output: Dict[str, Any],
        correct_answer: Dict[str, Any],
        **kwargs: object,
    ) -> EvaluationResult:
        """Evaluate with weighted scoring."""
        total_weight = sum(self.weights.values())
        weighted_score = 0.0
        criteria_results = {}

        for criterion, weight in self.weights.items():
            expected = correct_answer.get(criterion)
            actual = system_output.get(criterion)

            if expected == actual:
                score = 1.0
            elif expected is None or actual is None:
                score = 0.0
            elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                # Numeric: partial credit based on closeness
                diff = abs(expected - actual) / max(abs(expected), 1)
                score = max(0, 1 - diff)
            else:
                score = 0.0

            weighted_score += score * weight
            criteria_results[criterion] = {"expected": expected, "actual": actual, "score": score}

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return EvaluationResult(
            is_correct=final_score >= 0.9,  # Threshold for "correct"
            score=final_score,
            metadata={"criteria": criteria_results},
        )


def demonstrate_weighted_scoring() -> None:
    """Show weighted scoring evaluation."""
    print("Part 5: Weighted Scoring")
    print("-" * 50)

    evaluator = WeightedEvaluator(
        weights={"accuracy": 0.5, "completeness": 0.3, "format": 0.2}
    )

    expected = {"accuracy": 100, "completeness": 100, "format": "json"}

    test_cases = [
        {"accuracy": 100, "completeness": 100, "format": "json"},  # Perfect
        {"accuracy": 95, "completeness": 80, "format": "json"},  # Partial
        {"accuracy": 100, "completeness": 100, "format": "text"},  # Wrong format
    ]

    for i, output in enumerate(test_cases):
        result = evaluator.evaluate(output, expected)
        status = "PASS" if result.is_correct else "FAIL"
        print(f"  Test {i + 1} [{status}]:")
        print(f"    Output: {output}")
        print(f"    Score: {result.score:.2f}")
    print()


# =============================================================================
# Part 6: Evaluation Guidelines
# =============================================================================

def demonstrate_guidelines() -> None:
    """Show evaluation best practices."""
    print("Part 6: Evaluation Guidelines")
    print("-" * 50)

    guidelines = [
        (
            "Match evaluator to task type",
            "Exact match for factual QA, numeric tolerance for math, semantic for open-ended.",
        ),
        (
            "Define clear correctness criteria",
            "Specify what counts as correct before evaluation, not after.",
        ),
        (
            "Use appropriate tolerances",
            "Too strict = false negatives, too loose = false positives.",
        ),
        (
            "Track metadata for debugging",
            "Include why something failed, not just that it failed.",
        ),
        (
            "Consider partial credit",
            "Binary scoring loses information; gradual scores reveal trends.",
        ),
        (
            "Normalize before comparing",
            "Lowercase, strip whitespace, standardize formats.",
        ),
    ]

    for i, (title, description) in enumerate(guidelines, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
    print()


def main() -> None:
    """Demonstrate accuracy evaluation patterns."""
    print("Accuracy Evaluation Patterns")
    print("=" * 50)
    print()

    demonstrate_evaluation_result()
    demonstrate_exact_match()
    demonstrate_numeric_tolerance()
    demonstrate_custom_evaluator()
    demonstrate_weighted_scoring()
    demonstrate_guidelines()

    print("Key Takeaways")
    print("-" * 50)
    print("1. EvaluationResult captures correctness, score, and metadata")
    print("2. ExactMatchEvaluator for precise string matching")
    print("3. NumericToleranceEvaluator handles floating-point comparisons")
    print("4. Custom evaluators implement the IEvaluator interface")
    print("5. Weighted scoring enables multi-criteria evaluation")
    print("6. Include metadata for debugging and analysis")
    print()
    print("Next: See benchmark_harness.py for batch evaluation")


if __name__ == "__main__":
    main()
