"""Enhanced JIT API Demonstration.

This script demonstrates nested operator analysis using the enhanced JIT system.
It shows how the hierarchical analysis in the AutoGraphBuilder correctly identifies
dependencies between operators while respecting their hierarchical relationships.

This is a key component of both the @jit decorator (which uses trace-based graph building)
and the @structural_jit decorator (which uses structure-based graph building).

For a comprehensive explanation of the relationship between these approaches,
see docs/xcs/JIT_OVERVIEW.md.

To run:
    uv run python src/ember/examples/xcs/enhanced_jit_example.py
"""

import logging

from ember.xcs.graph.xcs_graph import XCSGraph

# For tracing with the autograph module
from ember.xcs.tracer.autograph import AutoGraphBuilder
from ember.xcs.tracer.xcs_tracing import TraceRecord

###############################################################################
# Mock Execution Setup
###############################################################################


def build_graph_example() -> None:
    """Demonstrate graph building from trace records with nested operators."""

    # Create trace records that represent a nested execution pattern:
    # Top level pipeline contains nested operators
    records = [
        TraceRecord(
            operator_name="MainPipeline",
            node_id="pipeline1",
            inputs={"query": "What is machine learning?"},
            outputs={"result": "pipeline_result"},
            timestamp=1.0,
        ),
        TraceRecord(
            operator_name="Refiner",
            node_id="refiner1",
            inputs={"query": "What is machine learning?"},
            outputs={"refined_query": "Improved: What is machine learning?"},
            timestamp=1.1,  # Executed during pipeline
        ),
        TraceRecord(
            operator_name="Ensemble",
            node_id="ensemble1",
            inputs={"refined_query": "Improved: What is machine learning?"},
            outputs={"responses": ["Answer 1", "Answer 2"]},
            timestamp=1.2,  # Executed during pipeline
        ),
        TraceRecord(
            operator_name="Generator1",
            node_id="gen1",
            inputs={"refined_query": "Improved: What is machine learning?"},
            outputs={"answer": "Answer 1"},
            timestamp=1.21,  # Executed during ensemble
        ),
        TraceRecord(
            operator_name="Generator2",
            node_id="gen2",
            inputs={"refined_query": "Improved: What is machine learning?"},
            outputs={"answer": "Answer 2"},
            timestamp=1.22,  # Executed during ensemble
        ),
        TraceRecord(
            operator_name="Aggregator",
            node_id="agg1",
            inputs={"responses": ["Answer 1", "Answer 2"]},
            outputs={"final_answer": "Machine learning is..."},
            timestamp=1.3,  # Executed during pipeline
        ),
        TraceRecord(
            operator_name="NextQuery",
            node_id="next1",
            inputs={
                "previous_result": "Machine learning is...",
                "answer": "Answer 1",  # This creates a data dependency with Generator1's output
            },
            outputs={"new_query": "Tell me more about supervised learning"},
            timestamp=2.0,  # Executed after pipeline
        ),
    ]

    # Build graph with standard dependency analysis (no hierarchy awareness)
    basic_builder = AutoGraphBuilder()
    # Disable hierarchy analysis by providing empty map
    basic_builder._build_hierarchy_map = lambda records: {}
    basic_graph = basic_builder.build_graph(records)

    # Build graph with hierarchical dependency analysis
    enhanced_builder = AutoGraphBuilder()
    # Explicitly define hierarchy to demonstrate the point more clearly
    hierarchy_map = {
        "pipeline1": ["refiner1", "ensemble1", "agg1"],
        "ensemble1": ["gen1", "gen2"],
    }
    enhanced_builder._build_hierarchy_map = lambda records: hierarchy_map
    enhanced_graph = enhanced_builder.build_graph(records)

    # Print the results
    print("\n--- BASIC GRAPH (without hierarchical analysis) ---")
    print_graph_dependencies(basic_graph)

    print("\n--- ENHANCED GRAPH (with hierarchical analysis) ---")
    print_graph_dependencies(enhanced_graph)

    # Print the key differences
    print("\n--- KEY DIFFERENCES (EXPECTED) ---")
    print("In a correctly implemented hierarchical analysis:")
    print("1. NextQuery should NOT depend on Generator1 and Generator2 directly")
    print("   (since they are nested inside Ensemble)")
    print("2. Aggregator should NOT depend on Generator1 and Generator2 directly")
    print("   (should depend only on Ensemble)")

    # Analyze both graphs to find actual differences
    basic_deps = {}
    enhanced_deps = {}

    for node_id, node in basic_graph.nodes.items():
        basic_deps[node_id] = list(node.inbound_edges)

    for node_id, node in enhanced_graph.nodes.items():
        enhanced_deps[node_id] = list(node.inbound_edges)

    print("\n--- ACTUAL DIFFERENCES DETECTED ---")
    for node_id in basic_deps.keys():
        basic_deps_set = set(basic_deps.get(node_id, []))
        enhanced_deps_set = set(enhanced_deps.get(node_id, []))

        if basic_deps_set != enhanced_deps_set:
            print(f"Node: {node_id}")
            print(
                f"  Basic dependencies: {', '.join(basic_deps_set) if basic_deps_set else 'None'}"
            )
            print(
                f"  Enhanced dependencies: {', '.join(enhanced_deps_set) if enhanced_deps_set else 'None'}"
            )
            print(
                f"  Removed in enhanced: {', '.join(basic_deps_set - enhanced_deps_set) if basic_deps_set - enhanced_deps_set else 'None'}"
            )
            print(
                f"  Added in enhanced: {', '.join(enhanced_deps_set - basic_deps_set) if enhanced_deps_set - basic_deps_set else 'None'}"
            )
            print()


def print_graph_dependencies(graph: XCSGraph) -> None:
    """Print the dependencies in a graph."""
    for node_id, node in graph.nodes.items():
        if node.inbound_edges:
            print(f"{node_id} depends on: {', '.join(node.inbound_edges)}")
        else:
            print(f"{node_id}: No dependencies")


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run the nested operator analysis demonstration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Enhanced JIT Example - Testing Hierarchical Dependency Analysis")
    print(
        "This demonstrates how the enhanced JIT system correctly handles nested operators.\n"
    )

    build_graph_example()


if __name__ == "__main__":
    main()
