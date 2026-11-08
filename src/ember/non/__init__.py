"""NON: a one-line notation for networks of model calls, compiled to operators.

Example:
    >>> from ember.non import build_graph
    >>> graph = build_graph("5E@openai/gpt-5.5(reasoning_effort=low), 1M")
    >>> [type(stage).__name__ for stage in graph.stages()]
    ['ModelStage', 'VoteStage']
"""

from ember.non.grammar import ModelStage, Params, Role, Stage, VoteStage, parse
from ember.non.graph import (
    Answer,
    NonGraph,
    Sampled,
    StageRecord,
    Verified,
    Voted,
    build_graph,
)
from ember.non.plan import Plan, StagePlan

__all__ = [
    "Answer",
    "ModelStage",
    "NonGraph",
    "Params",
    "Plan",
    "Role",
    "Sampled",
    "Stage",
    "StagePlan",
    "StageRecord",
    "Verified",
    "VoteStage",
    "Voted",
    "build_graph",
    "parse",
]
