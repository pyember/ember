#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jit_example.py

Demonstration of three variants of a "LargeEnsemble" operator (using Avior's non.py + operator_registry):
  1) BaselineEnsemble (no parallelism, fully eager)
  2) ParallelEnsemble (concurrency via to_plan)
  3) JITEnsemble (parallel but JIT-traced, reusing the concurrency plan)

We then measure the total and per-query times for each approach.
All rely on the existing code in non.py, which wraps EnsembleOperator from operator_registry.
"""

import time
import logging
from typing import List

from prettytable import PrettyTable

# Avior imports
from src.avior.core.configs.config import initialize_system
from src.avior.registry.model.registry.model_registry import GLOBAL_MODEL_REGISTRY
from src.avior.core.operator_graph_runner import OperatorGraphRunner
from src.avior.core.operator_graph import OperatorGraph
from src.avior.core.tracer_decorator import jit

# The existing 'Ensemble' class from non.py
from src.avior.registry.non import Ensemble, EnsembleInputs


###############################################################################
# 1) BaselineEnsemble - Eager Only (no concurrency)
###############################################################################
class BaselineEnsemble(Ensemble):
    """
    Subclass of `non.Ensemble` that overrides concurrency
    by returning `None` in to_plan(...), forcing a purely eager approach.
    """

    def to_plan(self, inputs: EnsembleInputs):
        # Disables concurrency => no tasks => fully serial
        return None


###############################################################################
# 2) ParallelEnsemble - Standard concurrency (no changes)
###############################################################################
class ParallelEnsemble(Ensemble):
    """
    Inherits directly from `non.Ensemble`.
    That class already uses 'EnsembleOperator' under the hood,
    which returns a concurrency plan in to_plan(...).
    
    No overrides => concurrency is enabled if the runner's max_workers > 1.
    """
    pass


###############################################################################
# 3) JITEnsemble - Parallel + JIT Tracing
###############################################################################
@jit(sample_input={"query": "JIT Warmup Sample"})
class JITEnsemble(ParallelEnsemble):
    """
    Same concurrency approach as ParallelEnsemble, but decorated with @jit.
    After the first call (or on __init__ if sample_input is specified),
    a concurrency plan is traced & cached, so subsequent calls skip overhead.
    """
    pass


###############################################################################
# 4) Main demonstration
###############################################################################
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1) Initialize the Avior system
    initialize_system(registry=GLOBAL_MODEL_REGISTRY)
    # Optionally set: CONFIG.set("models", "openai_api_key", "<YOUR-KEY>")

    # 2) Create one instance of each style operator
    #    We'll do 100 sub-calls (num_units=100).
    #    Omit max_tokens to keep it minimal, pass any model_name you want.
    model_name = "openai:gpt-4o-mini"
    temperature = 0.7
    NUM_UNITS = 10

    baseline_op = BaselineEnsemble(
        num_units=NUM_UNITS,
        model_name=model_name,
        temperature=temperature,
    )

    parallel_op = ParallelEnsemble(
        num_units=NUM_UNITS,
        model_name=model_name,
        temperature=temperature,
    )

    jit_op = JITEnsemble(
        num_units=NUM_UNITS,
        model_name=model_name,
        temperature=temperature,
    )

    # 3) We'll run multiple queries with one OperatorGraphRunner that has many workers (e.g. 32),
    #    maximizing concurrency for the parallel versions.
    runner = OperatorGraphRunner(max_workers=NUM_UNITS)

    queries = [
        "What is 2 + 2?",
        "Summarize quantum entanglement in simple terms.",
        "Longest river in Europe?",
        "Explain synergy in a business context.",
        "Who wrote Pride and Prejudice?",
    ]

    baseline_times: List[float] = []
    parallel_times: List[float] = []
    jit_times: List[float] = []

    # ########################################################################
    # # A) Baseline Ensemble: no concurrency
    # ########################################################################
    start_baseline_total = time.perf_counter()
    for q in queries:
        g = OperatorGraph()
        g.add_node(baseline_op, node_id="baseline_op")

        t1 = time.perf_counter()
        out = runner.run(g, {"query": q})
        t2 = time.perf_counter()

        baseline_times.append(t2 - t1)
        logging.info(f"[BASELINE] Query='{q}' => #responses={len(out.get('responses', []))} | time={t2 - t1:.4f}s")

    end_baseline_total = time.perf_counter()
    total_baseline_time = end_baseline_total - start_baseline_total

    # ########################################################################
    # # B) Parallel Ensemble
    # ########################################################################
    start_parallel_total = time.perf_counter()
    for q in queries:
        g = OperatorGraph()
        g.add_node(parallel_op, node_id="parallel_op")

        t1 = time.perf_counter()
        out = runner.run(g, {"query": q})
        t2 = time.perf_counter()

        parallel_times.append(t2 - t1)
        logging.info(f"[PARALLEL] Query='{q}' => #responses={len(out.get('responses', []))} | time={t2 - t1:.4f}s")

    end_parallel_total = time.perf_counter()
    total_parallel_time = end_parallel_total - start_parallel_total

    ########################################################################
    # C) JIT Parallel Ensemble
    ########################################################################
    start_jit_total = time.perf_counter()
    for q in queries:
        g = OperatorGraph()
        g.add_node(jit_op, node_id="jit_op")

        t1 = time.perf_counter()
        out = runner.run(g, {"query": q})
        t2 = time.perf_counter()

        jit_times.append(t2 - t1)
        logging.info(f"[JIT] Query='{q}' => #responses={len(out.get('responses', []))} | time={t2 - t1:.4f}s")

    end_jit_total = time.perf_counter()
    total_jit_time = end_jit_total - start_jit_total

    ########################################################################
    # 4) Print timing summary
    ########################################################################
    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["Query #", "Baseline (s)", "Parallel (s)", "JIT (s)"]
    for i, q in enumerate(queries):
        table.add_row([
            i+1,
            f"{baseline_times[i]:.4f}",
            f"{parallel_times[i]:.4f}",
            f"{jit_times[i]:.4f}"
        ])

    print("\nTiming Results for Baseline vs. Parallel vs. JIT (100 LM calls each, max_workers=32):")
    print(table)

    print(f"\nTotal Baseline time: {total_baseline_time:.4f}s")
    print(f"Total Parallel time: {total_parallel_time:.4f}s")
    print(f"Total JIT time:      {total_jit_time:.4f}s")

    avg_base = sum(baseline_times) / len(baseline_times)
    avg_par = sum(parallel_times) / len(parallel_times)
    avg_jit = sum(jit_times) / len(jit_times)

    print(f"\nAverage per-query Baseline: {avg_base:.4f}s")
    print(f"Average per-query Parallel: {avg_par:.4f}s")
    print(f"Average per-query JIT:      {avg_jit:.4f}s")


if __name__ == "__main__":
    main()