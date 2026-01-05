"""Benchmark harness for systematic LLM evaluation.

This example demonstrates:
- Batch evaluation patterns
- Pipeline evaluators with transforms
- Benchmark dataset handling
- Results aggregation and reporting
- Performance metrics tracking

Run with:
    python examples/10_evaluation_suite/benchmark_harness.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ember.utils.eval import (
    BatchEvaluationSummary,
    EvaluationResult,
    ExactMatchEvaluator,
    IEvaluator,
    PipelineEvaluator,
    evaluate_batch,
    evaluate_batch_with_summary,
)


# =============================================================================
# Part 1: Batch Evaluation Basics
# =============================================================================

def demonstrate_batch_evaluation() -> None:
    """Show basic batch evaluation."""
    print("Part 1: Batch Evaluation Basics")
    print("-" * 50)

    evaluator = ExactMatchEvaluator()

    # Sample QA dataset
    system_outputs = [
        "Paris",
        "Berlin",
        "Tokyo",
        "London",
        "Rome",
    ]
    correct_answers = [
        "Paris",
        "Berlin",
        "Tokyo",
        "Madrid",  # Wrong
        "Rome",
    ]

    # Evaluate batch
    results = evaluate_batch(evaluator, system_outputs, correct_answers)

    print("Individual results:")
    for i, (output, expected, result) in enumerate(
        zip(system_outputs, correct_answers, results)
    ):
        status = "PASS" if result.is_correct else "FAIL"
        print(f"  {i + 1}. [{status}] '{output}' vs '{expected}'")

    # Summary
    summary = evaluate_batch_with_summary(evaluator, system_outputs, correct_answers)
    print(f"\nSummary:")
    print(f"  Accuracy: {summary.accuracy:.1%}")
    print(f"  Mean score: {summary.mean_score:.2f}")
    print()


# =============================================================================
# Part 2: Pipeline Evaluators
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def extract_final_answer(text: str) -> str:
    """Extract final answer from verbose output."""
    # Look for "Answer: X" pattern
    if "answer:" in text.lower():
        parts = text.lower().split("answer:")
        if len(parts) > 1:
            return parts[-1].strip().split()[0] if parts[-1].strip() else text
    return text


def demonstrate_pipeline_evaluation() -> None:
    """Show pipeline evaluators with transforms."""
    print("Part 2: Pipeline Evaluators")
    print("-" * 50)

    # Create pipeline with transforms
    pipeline = PipelineEvaluator(
        transforms=[normalize_text, extract_final_answer],
        evaluator=ExactMatchEvaluator(),
    )

    test_cases = [
        ("The answer is: PARIS", "paris"),
        ("After analysis, Answer: Berlin", "berlin"),
        ("Tokyo", "tokyo"),
    ]

    print("Pipeline: normalize -> extract_answer -> exact_match")
    for output, expected in test_cases:
        result = pipeline.evaluate(output, expected)
        status = "PASS" if result.is_correct else "FAIL"
        print(f"  [{status}] '{output}' -> '{expected}'")
    print()


# =============================================================================
# Part 3: Benchmark Dataset
# =============================================================================

@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    id: str
    input: str
    expected_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """A benchmark dataset with samples."""

    name: str
    samples: List[BenchmarkSample] = field(default_factory=list)
    description: str = ""

    def add_sample(
        self,
        id: str,
        input: str,
        expected: str,
        **metadata: Any,
    ) -> "BenchmarkDataset":
        """Add a sample to the dataset."""
        self.samples.append(BenchmarkSample(id, input, expected, metadata))
        return self

    def __len__(self) -> int:
        return len(self.samples)


def create_sample_benchmark() -> BenchmarkDataset:
    """Create a sample benchmark dataset."""
    dataset = BenchmarkDataset(
        name="capital_cities",
        description="Geography questions about capital cities",
    )

    dataset.add_sample("q1", "What is the capital of France?", "Paris", category="europe")
    dataset.add_sample("q2", "What is the capital of Japan?", "Tokyo", category="asia")
    dataset.add_sample("q3", "What is the capital of Brazil?", "Brasilia", category="south_america")
    dataset.add_sample("q4", "What is the capital of Australia?", "Canberra", category="oceania")
    dataset.add_sample("q5", "What is the capital of Egypt?", "Cairo", category="africa")

    return dataset


def demonstrate_benchmark_dataset() -> None:
    """Show benchmark dataset handling."""
    print("Part 3: Benchmark Dataset")
    print("-" * 50)

    dataset = create_sample_benchmark()
    print(f"Dataset: {dataset.name}")
    print(f"Description: {dataset.description}")
    print(f"Samples: {len(dataset)}")
    print("\nSample entries:")
    for sample in dataset.samples[:3]:
        print(f"  {sample.id}: {sample.input}")
        print(f"    Expected: {sample.expected_output}")
        print(f"    Metadata: {sample.metadata}")
    print()


# =============================================================================
# Part 4: Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""

    dataset_name: str
    summary: BatchEvaluationSummary
    per_sample: List[Dict[str, Any]]
    runtime_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Run benchmarks with evaluation."""

    def __init__(
        self,
        evaluator: IEvaluator,
        system_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.evaluator = evaluator
        self.system_fn = system_fn or (lambda x: x)

    def run(
        self,
        dataset: BenchmarkDataset,
        system_outputs: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run the benchmark."""
        start_time = time.perf_counter()

        # Get system outputs
        if system_outputs is None:
            system_outputs = [self.system_fn(s.input) for s in dataset.samples]

        expected = [s.expected_output for s in dataset.samples]

        # Evaluate
        summary = evaluate_batch_with_summary(
            self.evaluator,
            system_outputs,
            expected,
        )

        # Per-sample details
        per_sample = []
        for sample, output, result in zip(dataset.samples, system_outputs, summary.results):
            per_sample.append({
                "id": sample.id,
                "input": sample.input,
                "expected": sample.expected_output,
                "output": output,
                "is_correct": result.is_correct,
                "score": result.score,
                "metadata": sample.metadata,
            })

        runtime = time.perf_counter() - start_time

        return BenchmarkResult(
            dataset_name=dataset.name,
            summary=summary,
            per_sample=per_sample,
            runtime_seconds=runtime,
        )


def demonstrate_benchmark_runner() -> None:
    """Show benchmark runner in action."""
    print("Part 4: Benchmark Runner")
    print("-" * 50)

    dataset = create_sample_benchmark()

    # Simulated system outputs (with some errors)
    system_outputs = [
        "Paris",      # Correct
        "Tokyo",      # Correct
        "Rio",        # Wrong (should be Brasilia)
        "Canberra",   # Correct
        "Cairo",      # Correct
    ]

    runner = BenchmarkRunner(evaluator=ExactMatchEvaluator())
    result = runner.run(dataset, system_outputs)

    print(f"Benchmark: {result.dataset_name}")
    print(f"Accuracy: {result.summary.accuracy:.1%}")
    print(f"Runtime: {result.runtime_seconds * 1000:.1f}ms")
    print("\nPer-sample results:")
    for sample in result.per_sample:
        status = "PASS" if sample["is_correct"] else "FAIL"
        print(f"  [{status}] {sample['id']}: '{sample['output']}' (expected: '{sample['expected']}')")
    print()


# =============================================================================
# Part 5: Results Aggregation
# =============================================================================

@dataclass
class AggregatedResults:
    """Aggregated results across multiple benchmarks or categories."""

    total_samples: int
    total_correct: int
    overall_accuracy: float
    by_category: Dict[str, Dict[str, float]]


def aggregate_by_category(result: BenchmarkResult) -> AggregatedResults:
    """Aggregate results by category metadata."""
    category_stats: Dict[str, Dict[str, int]] = {}

    for sample in result.per_sample:
        category = sample["metadata"].get("category", "uncategorized")
        if category not in category_stats:
            category_stats[category] = {"total": 0, "correct": 0}

        category_stats[category]["total"] += 1
        if sample["is_correct"]:
            category_stats[category]["correct"] += 1

    by_category = {}
    for cat, stats in category_stats.items():
        by_category[cat] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "total": stats["total"],
            "correct": stats["correct"],
        }

    return AggregatedResults(
        total_samples=len(result.per_sample),
        total_correct=sum(1 for s in result.per_sample if s["is_correct"]),
        overall_accuracy=result.summary.accuracy,
        by_category=by_category,
    )


def demonstrate_aggregation() -> None:
    """Show results aggregation."""
    print("Part 5: Results Aggregation")
    print("-" * 50)

    dataset = create_sample_benchmark()
    system_outputs = ["Paris", "Tokyo", "Rio", "Canberra", "Cairo"]

    runner = BenchmarkRunner(evaluator=ExactMatchEvaluator())
    result = runner.run(dataset, system_outputs)

    aggregated = aggregate_by_category(result)

    print(f"Overall: {aggregated.total_correct}/{aggregated.total_samples} "
          f"({aggregated.overall_accuracy:.1%})")
    print("\nBy category:")
    for category, stats in aggregated.by_category.items():
        print(f"  {category}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    print()


# =============================================================================
# Part 6: Reporting
# =============================================================================

def generate_report(result: BenchmarkResult) -> str:
    """Generate a text report from benchmark results."""
    lines = [
        f"Benchmark Report: {result.dataset_name}",
        "=" * 50,
        "",
        "Summary",
        "-" * 20,
        f"  Samples evaluated: {len(result.per_sample)}",
        f"  Accuracy: {result.summary.accuracy:.1%}",
        f"  Mean score: {result.summary.mean_score:.3f}",
        f"  Runtime: {result.runtime_seconds:.3f}s",
        "",
        "Failures",
        "-" * 20,
    ]

    failures = [s for s in result.per_sample if not s["is_correct"]]
    if failures:
        for f in failures:
            lines.append(f"  {f['id']}: expected '{f['expected']}', got '{f['output']}'")
    else:
        lines.append("  No failures!")

    return "\n".join(lines)


def demonstrate_reporting() -> None:
    """Show report generation."""
    print("Part 6: Report Generation")
    print("-" * 50)

    dataset = create_sample_benchmark()
    system_outputs = ["Paris", "Tokyo", "Rio", "Canberra", "Cairo"]

    runner = BenchmarkRunner(evaluator=ExactMatchEvaluator())
    result = runner.run(dataset, system_outputs)

    report = generate_report(result)
    print(report)
    print()


def main() -> None:
    """Demonstrate benchmark harness patterns."""
    print("Benchmark Harness Patterns")
    print("=" * 50)
    print()

    demonstrate_batch_evaluation()
    demonstrate_pipeline_evaluation()
    demonstrate_benchmark_dataset()
    demonstrate_benchmark_runner()
    demonstrate_aggregation()
    demonstrate_reporting()

    print("Key Takeaways")
    print("-" * 50)
    print("1. evaluate_batch processes multiple samples efficiently")
    print("2. PipelineEvaluator chains transforms before evaluation")
    print("3. BenchmarkDataset organizes samples with metadata")
    print("4. BenchmarkRunner provides timing and per-sample details")
    print("5. Aggregation reveals performance by category")
    print("6. Reports make results actionable")
    print()
    print("Next: See consistency_testing.py for reliability testing")


if __name__ == "__main__":
    main()
