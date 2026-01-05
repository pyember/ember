"""Consistency testing for LLM output reliability.

This example demonstrates:
- Testing output stability across multiple runs
- Variance analysis
- Agreement metrics
- Temperature sensitivity testing
- Stateful evaluation patterns

Run with:
    python examples/10_evaluation_suite/consistency_testing.py
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ember.utils.eval import EvaluationResult, IEvaluator, IStatefulEvaluator


# =============================================================================
# Part 1: Basic Consistency Check
# =============================================================================

@dataclass
class ConsistencyResult:
    """Result of consistency testing."""

    responses: List[str]
    unique_responses: int
    most_common: str
    most_common_count: int
    agreement_rate: float
    is_consistent: bool


def check_consistency(responses: List[str], threshold: float = 0.8) -> ConsistencyResult:
    """Check consistency of multiple responses."""
    counter = Counter(responses)
    most_common, most_common_count = counter.most_common(1)[0]

    agreement_rate = most_common_count / len(responses) if responses else 0

    return ConsistencyResult(
        responses=responses,
        unique_responses=len(counter),
        most_common=most_common,
        most_common_count=most_common_count,
        agreement_rate=agreement_rate,
        is_consistent=agreement_rate >= threshold,
    )


def demonstrate_basic_consistency() -> None:
    """Show basic consistency checking."""
    print("Part 1: Basic Consistency Check")
    print("-" * 50)

    # Simulate multiple runs
    test_cases = [
        (["Paris", "Paris", "Paris", "Paris", "Paris"], "Highly consistent"),
        (["Paris", "Paris", "Paris", "London", "Paris"], "Mostly consistent"),
        (["Paris", "London", "Berlin", "Tokyo", "Madrid"], "Inconsistent"),
    ]

    for responses, description in test_cases:
        result = check_consistency(responses, threshold=0.8)
        status = "PASS" if result.is_consistent else "FAIL"
        print(f"  [{status}] {description}")
        print(f"    Responses: {responses}")
        print(f"    Agreement: {result.agreement_rate:.1%} ({result.most_common_count}/{len(responses)})")
        print(f"    Most common: '{result.most_common}'")
    print()


# =============================================================================
# Part 2: Variance Analysis
# =============================================================================

@dataclass
class VarianceAnalysis:
    """Analysis of response variance."""

    mean_length: float
    std_length: float
    length_cv: float  # Coefficient of variation
    unique_tokens: int
    semantic_diversity: float


def analyze_variance(responses: List[str]) -> VarianceAnalysis:
    """Analyze variance in responses."""
    if not responses:
        return VarianceAnalysis(0, 0, 0, 0, 0)

    # Length statistics
    lengths = [len(r) for r in responses]
    mean_length = sum(lengths) / len(lengths)
    variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
    std_length = variance ** 0.5
    length_cv = std_length / mean_length if mean_length > 0 else 0

    # Token diversity
    all_tokens = set()
    for r in responses:
        all_tokens.update(r.lower().split())

    # Simple semantic diversity (based on unique responses)
    unique_count = len(set(responses))
    semantic_diversity = unique_count / len(responses) if responses else 0

    return VarianceAnalysis(
        mean_length=mean_length,
        std_length=std_length,
        length_cv=length_cv,
        unique_tokens=len(all_tokens),
        semantic_diversity=semantic_diversity,
    )


def demonstrate_variance_analysis() -> None:
    """Show variance analysis."""
    print("Part 2: Variance Analysis")
    print("-" * 50)

    # Low variance (consistent)
    low_variance = [
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is Paris.",
    ]

    # High variance (inconsistent)
    high_variance = [
        "Paris is the capital.",
        "The capital city of the French Republic is Paris, a major European city.",
        "France's capital? That would be Paris.",
    ]

    for name, responses in [("Low variance", low_variance), ("High variance", high_variance)]:
        analysis = analyze_variance(responses)
        print(f"  {name}:")
        print(f"    Mean length: {analysis.mean_length:.1f} chars")
        print(f"    Std deviation: {analysis.std_length:.1f}")
        print(f"    Coefficient of variation: {analysis.length_cv:.2f}")
        print(f"    Semantic diversity: {analysis.semantic_diversity:.2f}")
        print()


# =============================================================================
# Part 3: Agreement Metrics
# =============================================================================

def pairwise_agreement(responses: List[str]) -> float:
    """Calculate pairwise agreement rate."""
    if len(responses) < 2:
        return 1.0

    agreements = 0
    total_pairs = 0

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            if responses[i] == responses[j]:
                agreements += 1
            total_pairs += 1

    return agreements / total_pairs if total_pairs > 0 else 0


def fleiss_kappa_simple(responses_matrix: List[List[str]]) -> float:
    """Simplified Fleiss' kappa for inter-rater reliability.

    Args:
        responses_matrix: List of [response1, response2, ...] for each item.
    """
    if not responses_matrix:
        return 0.0

    # Count categories
    all_categories = set()
    for responses in responses_matrix:
        all_categories.update(responses)

    n_items = len(responses_matrix)
    n_raters = len(responses_matrix[0]) if responses_matrix else 0

    if n_raters < 2:
        return 1.0

    # Calculate P_e (expected agreement by chance)
    category_counts = Counter()
    for responses in responses_matrix:
        category_counts.update(responses)

    total_ratings = n_items * n_raters
    p_e = sum((count / total_ratings) ** 2 for count in category_counts.values())

    # Calculate P_o (observed agreement)
    p_o_sum = 0
    for responses in responses_matrix:
        counter = Counter(responses)
        p_o_sum += sum(c * (c - 1) for c in counter.values())

    p_o = p_o_sum / (n_items * n_raters * (n_raters - 1)) if n_raters > 1 else 1

    # Kappa
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def demonstrate_agreement_metrics() -> None:
    """Show agreement metrics."""
    print("Part 3: Agreement Metrics")
    print("-" * 50)

    # Perfect agreement
    perfect = ["A", "A", "A", "A", "A"]
    print(f"  Perfect agreement: {perfect}")
    print(f"    Pairwise agreement: {pairwise_agreement(perfect):.2f}")

    # Partial agreement
    partial = ["A", "A", "A", "B", "B"]
    print(f"\n  Partial agreement: {partial}")
    print(f"    Pairwise agreement: {pairwise_agreement(partial):.2f}")

    # Multi-item Fleiss' kappa
    multi_item = [
        ["A", "A", "A"],  # Item 1: all agree
        ["A", "A", "B"],  # Item 2: partial
        ["A", "B", "C"],  # Item 3: disagree
    ]
    print(f"\n  Multi-item reliability:")
    print(f"    Items: {multi_item}")
    print(f"    Fleiss' kappa: {fleiss_kappa_simple(multi_item):.2f}")
    print()


# =============================================================================
# Part 4: Stateful Consistency Evaluator
# =============================================================================

@dataclass
class ConsistencyEvaluator(IStatefulEvaluator[str, str]):
    """Stateful evaluator that tracks consistency across samples."""

    threshold: float = 0.8
    _samples: List[Dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        system_output: str,
        correct_answer: str,
        **kwargs: object,
    ) -> None:
        """Add sample to tracking."""
        question_id = kwargs.get("question_id", len(self._samples))
        self._samples.append({
            "question_id": question_id,
            "output": system_output,
            "expected": correct_answer,
        })

    def compute(self) -> EvaluationResult:
        """Compute aggregated consistency."""
        if not self._samples:
            return EvaluationResult(is_correct=True, score=1.0)

        # Group by question
        by_question: Dict[Any, List[str]] = {}
        for sample in self._samples:
            qid = sample["question_id"]
            if qid not in by_question:
                by_question[qid] = []
            by_question[qid].append(sample["output"])

        # Calculate consistency per question
        consistencies = []
        for qid, outputs in by_question.items():
            result = check_consistency(outputs, self.threshold)
            consistencies.append(result.agreement_rate)

        avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0

        return EvaluationResult(
            is_correct=avg_consistency >= self.threshold,
            score=avg_consistency,
            metadata={
                "questions_evaluated": len(by_question),
                "per_question_consistency": consistencies,
            },
        )


def demonstrate_stateful_evaluation() -> None:
    """Show stateful consistency evaluation."""
    print("Part 4: Stateful Consistency Evaluator")
    print("-" * 50)

    evaluator = ConsistencyEvaluator(threshold=0.8)

    # Simulate multiple runs per question
    test_runs = [
        ("q1", "Paris", "Paris"),
        ("q1", "Paris", "Paris"),
        ("q1", "Paris", "Paris"),
        ("q2", "Tokyo", "Tokyo"),
        ("q2", "Kyoto", "Tokyo"),  # Inconsistent
        ("q2", "Tokyo", "Tokyo"),
    ]

    for qid, output, expected in test_runs:
        evaluator.update(output, expected, question_id=qid)

    result = evaluator.compute()
    print(f"  Overall consistency score: {result.score:.2f}")
    print(f"  Is consistent: {result.is_correct}")
    print(f"  Metadata: {result.metadata}")
    print()


# =============================================================================
# Part 5: Temperature Sensitivity
# =============================================================================

@dataclass
class TemperatureSensitivityResult:
    """Result of temperature sensitivity testing."""

    temperature: float
    responses: List[str]
    consistency: ConsistencyResult
    variance: VarianceAnalysis


def simulate_temperature_responses(
    base_response: str,
    temperature: float,
    num_samples: int = 5,
) -> List[str]:
    """Simulate responses at different temperatures."""
    responses = []
    variations = [
        base_response,
        base_response.upper(),
        base_response.lower(),
        f"The answer is {base_response}",
        f"{base_response} is the answer",
        f"I think it's {base_response}",
    ]

    for _ in range(num_samples):
        if temperature < 0.3:
            # Low temperature: mostly same response
            responses.append(base_response)
        elif temperature < 0.7:
            # Medium: some variation
            if random.random() < 0.7:
                responses.append(base_response)
            else:
                responses.append(random.choice(variations))
        else:
            # High: lots of variation
            responses.append(random.choice(variations))

    return responses


def demonstrate_temperature_sensitivity() -> None:
    """Show temperature sensitivity testing."""
    print("Part 5: Temperature Sensitivity")
    print("-" * 50)

    random.seed(42)  # Reproducible

    temperatures = [0.0, 0.5, 1.0]
    base_response = "Paris"

    print("Simulating responses at different temperatures:")
    for temp in temperatures:
        responses = simulate_temperature_responses(base_response, temp, num_samples=5)
        consistency = check_consistency(responses)
        variance = analyze_variance(responses)

        print(f"\n  Temperature {temp}:")
        print(f"    Responses: {responses}")
        print(f"    Agreement: {consistency.agreement_rate:.1%}")
        print(f"    Unique: {consistency.unique_responses}")
    print()


# =============================================================================
# Part 6: Consistency Testing Guidelines
# =============================================================================

def demonstrate_guidelines() -> None:
    """Show consistency testing best practices."""
    print("Part 6: Consistency Testing Guidelines")
    print("-" * 50)

    guidelines = [
        (
            "Test multiple runs per question",
            "5-10 runs reveal true consistency; single runs hide variance.",
        ),
        (
            "Set appropriate thresholds",
            "80% agreement for factual QA; lower for creative tasks.",
        ),
        (
            "Control temperature during testing",
            "Temperature=0 for reproducibility; higher for diversity testing.",
        ),
        (
            "Track by category",
            "Consistency may vary by question type or difficulty.",
        ),
        (
            "Monitor trends over time",
            "Model updates can affect consistency; track across versions.",
        ),
        (
            "Report both accuracy and consistency",
            "High accuracy with low consistency indicates unreliability.",
        ),
    ]

    for i, (title, description) in enumerate(guidelines, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
    print()


def main() -> None:
    """Demonstrate consistency testing patterns."""
    print("Consistency Testing Patterns")
    print("=" * 50)
    print()

    demonstrate_basic_consistency()
    demonstrate_variance_analysis()
    demonstrate_agreement_metrics()
    demonstrate_stateful_evaluation()
    demonstrate_temperature_sensitivity()
    demonstrate_guidelines()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Consistency testing reveals reliability beyond accuracy")
    print("2. Variance analysis quantifies response stability")
    print("3. Pairwise agreement and Fleiss' kappa measure inter-rater reliability")
    print("4. Stateful evaluators track consistency across multiple runs")
    print("5. Temperature sensitivity testing shows model stability")
    print("6. Report both accuracy and consistency for complete picture")
    print()
    print("Congratulations! You've completed the Ember examples suite.")


if __name__ == "__main__":
    main()
