"""Integration patterns with JAX and XCS transforms.

This example demonstrates:
- XCS transforms for orchestration workloads
- Hybrid tensor + orchestration patterns
- Analysis and traceability
- Best practices for JAX-compatible code

Run with:
    python examples/08_advanced_patterns/jax_xcs_integration.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ember.api import op
from ember.xcs import (
    Config,
    OpKind,
    Traceability,
    analyze_operations_v2,
    explain,
    jit,
    scan,
    vmap,
)


# =============================================================================
# Part 1: Understanding Operation Types
# =============================================================================

@op
def pure_computation(x: int) -> int:
    """Pure Python computation (no external calls)."""
    return x * x + x


@op
def orchestration_call(prompt: str) -> Dict[str, Any]:
    """Simulated orchestration call (would call LLM)."""
    return {"response": f"Response to: {prompt[:20]}..."}


def demonstrate_operation_analysis() -> None:
    """Show how XCS analyzes operations."""
    print("Part 1: Operation Analysis")
    print("-" * 50)

    # Analyze pure computation
    pure_decision = analyze_operations_v2(pure_computation)
    print("Pure computation analysis:")
    print(f"  Kind: {pure_decision.kind}")
    print(f"  JAX traceable: {pure_decision.jax_traceable}")
    print(f"  Effect risk: {pure_decision.effect_risk}")
    print()

    # The explain() function provides detailed insights
    print("Using explain() for detailed analysis:")
    explanation = explain(pure_computation)
    print(f"  {explanation}")
    print()


# =============================================================================
# Part 2: JIT for Orchestration
# =============================================================================

@jit
def jit_orchestration(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """JIT-compiled orchestration workflow.

    XCS automatically handles orchestration workloads by:
    1. Tracing the execution graph
    2. Identifying parallelization opportunities
    3. Using thread-based execution for non-tensor ops
    """
    results = []
    for item in items:
        processed = {
            "id": item.get("id"),
            "value": item.get("value", 0) * 2,
            "status": "processed",
        }
        results.append(processed)
    return results


def demonstrate_jit_orchestration() -> None:
    """Show JIT with orchestration workloads."""
    print("Part 2: JIT for Orchestration")
    print("-" * 50)

    items = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]

    # First call traces
    results = jit_orchestration(items)
    print("Processed items:")
    for r in results:
        print(f"  {r}")

    # Check stats
    stats = jit_orchestration.stats()  # type: ignore[attr-defined]
    print(f"\nJIT stats: status={stats.get('status')}, entries={stats.get('entries')}")
    print()


# =============================================================================
# Part 3: vmap for Batch Orchestration
# =============================================================================

@op
def process_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single item (orchestration-style)."""
    return {
        "input": item.get("value"),
        "output": item.get("value", 0) ** 2,
        "processed": True,
    }


def demonstrate_vmap_orchestration() -> None:
    """Show vmap with orchestration workloads."""
    print("Part 3: vmap for Batch Orchestration")
    print("-" * 50)

    # vmap automatically handles orchestration by parallel execution
    config = Config(parallel=True, max_workers=4)
    batch_process = vmap(process_single_item, config=config)

    items = [{"value": i} for i in range(5)]
    results = batch_process(items)

    print("Batch results:")
    for r in results:
        print(f"  {r}")
    print()


# =============================================================================
# Part 4: Scan for Stateful Iteration
# =============================================================================

def demonstrate_scan_patterns() -> None:
    """Show scan for stateful operations."""
    print("Part 4: Scan for Stateful Iteration")
    print("-" * 50)

    @scan
    def accumulate(state: Dict[str, int], item: int) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Accumulate running statistics."""
        new_count = state["count"] + 1
        new_sum = state["sum"] + item
        new_state = {"count": new_count, "sum": new_sum}
        output = {
            "item": item,
            "running_avg": new_sum / new_count,
        }
        return new_state, output

    initial = {"count": 0, "sum": 0}
    items = [10, 20, 30, 40, 50]

    final_state, outputs = accumulate(initial, items)

    print(f"Items: {items}")
    print(f"Final state: {final_state}")
    print("Running outputs:")
    for output in outputs:
        print(f"  {output}")
    print()


# =============================================================================
# Part 5: Hybrid Patterns
# =============================================================================

def demonstrate_hybrid_patterns() -> None:
    """Show hybrid tensor + orchestration patterns."""
    print("Part 5: Hybrid Patterns")
    print("-" * 50)

    # Pattern: Preprocess with pure functions, then orchestrate
    def hybrid_pipeline(data: List[int]) -> List[Dict[str, Any]]:
        """Pipeline combining pure computation and orchestration."""
        # Step 1: Pure tensor-like preprocessing
        preprocessed = [x ** 2 for x in data]

        # Step 2: Orchestration-style processing
        results = []
        for i, value in enumerate(preprocessed):
            results.append({
                "index": i,
                "original": data[i],
                "preprocessed": value,
                "category": "high" if value > 100 else "low",
            })

        return results

    # Wrap with JIT
    jit_pipeline = jit(hybrid_pipeline)

    data = [5, 10, 15, 20]
    results = jit_pipeline(data)

    print("Hybrid pipeline results:")
    for r in results:
        print(f"  {r}")
    print()


# =============================================================================
# Part 6: Best Practices
# =============================================================================

def demonstrate_best_practices() -> None:
    """Show best practices for XCS/JAX integration."""
    print("Part 6: Best Practices")
    print("-" * 50)

    practices = [
        (
            "Separate tensor ops from orchestration",
            "Keep pure computations separate for better optimization opportunities.",
        ),
        (
            "Use Config for parallelism control",
            "Config(parallel=True, max_workers=N) to control thread usage.",
        ),
        (
            "Analyze operations before optimization",
            "Use analyze_operations_v2() to understand operation characteristics.",
        ),
        (
            "Use vmap for independent batch items",
            "vmap parallelizes when items don't depend on each other.",
        ),
        (
            "Use scan for sequential dependencies",
            "scan handles operations that need state carried forward.",
        ),
        (
            "Check .stats() for optimization status",
            "Monitor cache hits and compilation status.",
        ),
    ]

    for i, (title, description) in enumerate(practices, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
        print()

    # Show analysis example
    print("Analysis example:")

    @jit
    def example_fn(x: int) -> int:
        return x * 2

    example_fn(5)  # Compile
    stats = example_fn.stats()  # type: ignore[attr-defined]
    print(f"  Status: {stats.get('status')}")
    print(f"  Optimized: {stats.get('optimized')}")
    print()


def main() -> None:
    """Demonstrate JAX/XCS integration patterns."""
    print("JAX/XCS Integration")
    print("=" * 50)
    print()

    demonstrate_operation_analysis()
    demonstrate_jit_orchestration()
    demonstrate_vmap_orchestration()
    demonstrate_scan_patterns()
    demonstrate_hybrid_patterns()
    demonstrate_best_practices()

    print("Key Takeaways")
    print("-" * 50)
    print("1. XCS adapts transforms for orchestration workloads")
    print("2. Use analyze_operations_v2() to understand op characteristics")
    print("3. JIT caches and optimizes repeated executions")
    print("4. vmap enables parallel batch processing")
    print("5. scan handles sequential operations with state")
    print("6. Separate pure computations from orchestration when possible")
    print()
    print("Next: Explore examples/09_practical_patterns/ for real-world patterns")


if __name__ == "__main__":
    main()
