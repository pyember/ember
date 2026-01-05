"""Compact NON graph toolkit.

The NON (Network of Networks) graph helpers provide a declarative notation
for building AI orchestration graphs. This module enables concise specification
of complex AI pipelines using a compact string notation.

Example:
    >>> from ember.non import build_graph, parse_flow_spec
    >>> # Create a 3-candidate ensemble with judge and verifier
    >>> graph = build_graph("3C->J->V")
    >>> print(describe_graph(graph))

The notation supports:
- Candidate stages (C): Generate candidate responses
- Judge stages (J): Evaluate and score candidates
- Aggregator stages (A): Combine multiple inputs
- Verifier stages (V): Validate outputs
"""

from __future__ import annotations

from .compact import (
    AggregatorStage,
    CandidateStage,
    CompactGraph,
    CompactNode,
    ExecutionPlan,
    FlowEndpoint,
    FlowSpec,
    JudgeStage,
    VerifierStage,
    build_graph,
    describe_graph,
    parse_flow_spec,
    parse_node_spec,
)
from .registry import DEFAULT_NODE_REGISTRY, NodeRegistry, NodeRole, NodeType

__all__ = [
    # Compact notation
    "ExecutionPlan",
    "CandidateStage",
    "AggregatorStage",
    "JudgeStage",
    "VerifierStage",
    "CompactGraph",
    "CompactNode",
    "FlowEndpoint",
    "FlowSpec",
    "build_graph",
    "describe_graph",
    "parse_flow_spec",
    "parse_node_spec",
    # Registry
    "NodeRegistry",
    "NodeType",
    "NodeRole",
    "DEFAULT_NODE_REGISTRY",
]
