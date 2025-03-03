"""Benchmark tests for operator execution performance in the Ember framework.

This module measures both sequential and concurrent execution latencies for a simple
operator (AddOneOperator) that increments an input by one. The tests report total, average,
minimum, maximum, median, and standard deviation metrics over multiple rounds, and also
compute key percentiles. Furthermore, robust assertions ensure that the median and deviations
remain within reasonable tolerances.

"""

import time
import statistics
import concurrent.futures
import logging
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel
from ember.core.registry.prompt_specification.specification import Specification
from ember.core.registry.operator.base.operator_base import Operator

logger: logging.Logger = logging.getLogger(__name__)


class DummyInput(BaseModel):
    """Represents a dummy input with an integer value."""

    value: int


class DummyOutput(BaseModel):
    """Represents a dummy output with an integer result."""

    result: int


class DummySpecification(Specification):
    """Specification for the dummy operator."""

    prompt_template: str = "{value}"
    input_model: Type[BaseModel] = DummyInput
    structured_output: Type[BaseModel] = DummyOutput
    check_all_placeholders: bool = False

    def validate_inputs(self, inputs: Dict[str, Any]) -> DummyInput:
        """Validates and converts a dictionary of inputs to a DummyInput instance.

        Args:
            inputs: A dictionary containing input data.

        Returns:
            A DummyInput instance with validated data.
        """
        return DummyInput(**inputs)

    def validate_output(self, output: Dict[str, Any]) -> DummyOutput:
        """Validates and converts a dictionary of outputs to a DummyOutput instance.

        Args:
            output: A dictionary containing output data.

        Returns:
            A DummyOutput instance with validated data.
        """
        if hasattr(output, "model_dump"):
            return DummyOutput(**output.model_dump())
        return DummyOutput(**output)


class AddOneOperator(Operator[DummyInput, DummyOutput]):
    """Operator that increments an input value by one."""

    specification: DummySpecification = DummySpecification()

    def forward(self, *, inputs: DummyInput) -> DummyOutput:
        """Computes the output by adding one to the input value.

        Args:
            inputs: A DummyInput instance.

        Returns:
            A DummyOutput instance with the incremented value.
        """
        return DummyOutput(result=inputs.value + 1)


def measure_sequential_performance(
    iterations: int, rounds: int = 5
) -> Tuple[List[float], Dict[str, float]]:
    """Measures sequential per-call execution latency over multiple rounds.

    Args:
        iterations: Number of operator invocations per round.
        rounds: Total number of rounds to perform.

    Returns:
        A tuple containing:
          - A list of average per-call latencies (in seconds) for each round.
          - A dictionary with overall statistics:
              * overall_avg: Mean latency over rounds.
              * min_latency: Minimum round-average latency.
              * max_latency: Maximum round-average latency.
              * std_latency: Standard deviation across rounds.
              * median_latency: Median round-average latency.
    """
    latencies: List[float] = []
    operator_instance: AddOneOperator = AddOneOperator()
    input_data: Dict[str, int] = {"value": 10}

    for round_index in range(rounds):
        start_time: float = time.perf_counter()
        for _ in range(iterations):
            result: DummyOutput = operator_instance(inputs=input_data)
            if result.result != 11:
                raise AssertionError(
                    f"Unexpected operator result: expected 11, got {result.result}"
                )
        end_time: float = time.perf_counter()
        total_time: float = end_time - start_time
        avg_latency: float = total_time / iterations
        latencies.append(avg_latency)
        logger.info(
            "Round %d: %d iterations in %.6f sec (Avg per call: %.6e sec)",
            round_index + 1,
            iterations,
            total_time,
            avg_latency,
        )

    stats: Dict[str, float] = {
        "overall_avg": statistics.mean(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "median_latency": statistics.median(latencies),
    }
    return latencies, stats


def measure_concurrent_performance(iterations: int, max_workers: int = 8) -> float:
    """Measures concurrent per-call execution latency using a thread pool.

    Args:
        iterations: Total number of operator invocations.
        max_workers: Maximum number of worker threads to use.

    Returns:
        The average per-call latency (in seconds) across all iterations.
    """
    operator_instance: AddOneOperator = AddOneOperator()
    input_data: Dict[str, int] = {"value": 10}

    def task() -> None:
        result: DummyOutput = operator_instance(inputs=input_data)
        if result.result != 11:
            raise AssertionError(
                f"Unexpected operator result: expected 11, got {result.result}"
            )

    start_time: float = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: List[concurrent.futures.Future] = [
            executor.submit(task) for _ in range(iterations)
        ]
        concurrent.futures.wait(futures)
    end_time: float = time.perf_counter()

    total_time: float = end_time - start_time
    avg_latency: float = total_time / iterations
    logger.info(
        "Concurrent: %d iterations with %d workers in %.6f sec (Avg per call: %.6e sec)",
        iterations,
        max_workers,
        total_time,
        avg_latency,
    )
    return avg_latency


def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Computes median and key percentiles from a list of latency measurements.

    Args:
        latencies: A list of latency measurements in seconds.

    Returns:
        A dictionary of computed percentiles:
          - "50th": 50th percentile (median)
          - "75th": 75th percentile
          - "90th": 90th percentile
          - "95th": 95th percentile
    """
    sorted_latencies: List[float] = sorted(latencies)
    count: int = len(sorted_latencies)
    percentiles: Dict[str, float] = {
        "50th": sorted_latencies[int(0.50 * (count - 1))],
        "75th": sorted_latencies[int(0.75 * (count - 1))],
        "90th": sorted_latencies[int(0.90 * (count - 1))],
        "95th": sorted_latencies[int(0.95 * (count - 1))],
    }
    return percentiles


def main() -> None:
    """Executes performance benchmarks for sequential and concurrent operator invocations."""
    rounds: int = 5
    iterations: int = 10000

    logger.info("Starting sequential performance benchmark...\n")
    seq_latencies, seq_stats = measure_sequential_performance(
        iterations=iterations, rounds=rounds
    )
    percentile_results: Dict[str, float] = compute_percentiles(seq_latencies)

    logger.info("\nSequential Benchmark Statistics:")
    logger.info("  Rounds:             %d", rounds)
    logger.info("  Overall Average:    %.6e sec", seq_stats["overall_avg"])
    logger.info("  Minimum Latency:    %.6e sec", seq_stats["min_latency"])
    logger.info("  Maximum Latency:    %.6e sec", seq_stats["max_latency"])
    logger.info("  Standard Deviation: %.6e sec", seq_stats["std_latency"])
    logger.info("  Median Latency:     %.6e sec", seq_stats["median_latency"])
    logger.info("  Percentiles:")
    for label, value in percentile_results.items():
        logger.info("    %s: %.6e sec", label, value)

    overall_avg: float = seq_stats["overall_avg"]
    std_dev: float = seq_stats["std_latency"]
    rel_std: float = std_dev / overall_avg if overall_avg > 0 else 0.0

    if overall_avg >= 1e-3:
        raise AssertionError(
            f"Sequential average latency {overall_avg:.6e} is too high."
        )
    if rel_std >= 0.30:
        raise AssertionError(
            f"Relative standard deviation {rel_std:.2f} is too high, indicating unstable measurements."
        )
    if seq_stats["max_latency"] >= 5 * overall_avg:
        raise AssertionError(
            f"Maximum sequential latency {seq_stats['max_latency']:.6e} is more than 5x the average {overall_avg:.6e}."
        )

    logger.info("\nStarting concurrent performance benchmark...\n")
    concur_avg: float = measure_concurrent_performance(
        iterations=iterations, max_workers=8
    )

    if (concur_avg / overall_avg) >= 2.0:
        raise AssertionError(
            f"Concurrent average latency {concur_avg:.6e} is more than twice the sequential average {overall_avg:.6e}."
        )

    logger.info("\nConcurrent Benchmark Average Latency: %.6e sec", concur_avg)
    logger.info("\nAll performance tests passed within defined tolerances.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
