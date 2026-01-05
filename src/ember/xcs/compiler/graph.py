"""Immutable graph structures for the XCS compiler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ember.xcs.errors import XCSError
from ember.xcs.utils.pytree import StaticWrapper, ensure_pytree_compatible


@dataclass(frozen=True, slots=True)
class IRNode:
    """A single computation node inside an IR graph.

    The dataclass accepts both ``id`` and ``node_id`` keyword arguments for
    backwards compatibility with the legacy implementation.
    """

    id: str
    operator: Any
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        node_id: Optional[str] = None,
        operator: Any,
        inputs: Sequence[str],
        outputs: Sequence[str],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        node_id = node_id or id
        if node_id is None:
            raise ValueError("IRNode requires an id")
        object.__setattr__(self, "id", node_id)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "outputs", tuple(outputs))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def node_id(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class IRGraph:
    """Immutable computation graph built from traced Python execution."""

    nodes: Mapping[str, IRNode] = field(default_factory=dict)
    edges: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    output_index: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_index:
            index: Dict[str, str] = {}
            for node in self.nodes.values():
                for output in node.outputs:
                    index[output] = node.id
            object.__setattr__(self, "output_index", index)
        if self.edges:
            normalized = {k: tuple(v) for k, v in self.edges.items()}
            object.__setattr__(self, "edges", normalized)

    def get_dependencies(self, node_id: str) -> Set[str]:
        node = self.nodes.get(node_id)
        if node is None:
            return set()
        deps: Set[str] = set()
        for input_name in node.inputs:
            producer = self.output_index.get(input_name)
            if producer:
                deps.add(producer)
        return deps

    def get_dependents(self, node_id: str) -> Tuple[str, ...]:
        return self.edges.get(node_id, tuple())

    def topological_order(self) -> List[str]:
        indegree: MutableMapping[str, int] = {name: 0 for name in self.nodes}
        for node in self.nodes.values():
            for input_name in node.inputs:
                producer = self.output_index.get(input_name)
                if producer:
                    indegree[node.id] += 1
        queue = deque([nid for nid, deg in indegree.items() if deg == 0])
        order: List[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for dependent in self.get_dependents(current):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if len(order) != len(self.nodes):
            missing = sorted(set(self.nodes) - set(order))
            raise XCSError(f"Cycle detected in IR graph; remaining nodes: {missing}")
        return order

    def add_node(self, node: IRNode) -> "IRGraph":
        new_nodes = dict(self.nodes)
        new_nodes[node.id] = node

        new_edges: Dict[str, Tuple[str, ...]] = dict(self.edges)
        new_index = dict(self.output_index)

        for input_name in node.inputs:
            producer = new_index.get(input_name)
            if producer:
                dependents = tuple(list(new_edges.get(producer, tuple())) + [node.id])
                new_edges[producer] = dependents

        for output in node.outputs:
            new_index[output] = node.id

        return IRGraph(nodes=new_nodes, edges=new_edges, output_index=new_index)


@dataclass(frozen=True, slots=True)
class ParallelismInfo:
    can_vmap: bool = False
    can_pmap: bool = False
    can_parallelize: bool = False
    estimated_speedup: float = 1.0
    is_pure: bool = True


@dataclass(frozen=True, slots=True)
class GraphParallelismAnalysis:
    node_info: Mapping[str, ParallelismInfo]
    parallel_groups: Sequence[Set[str]]
    vectorizable_chains: Sequence[Sequence[str]]
    estimated_speedup: float
    bottlenecks: Sequence[str]


def node_from_callable(
    node_id: str,
    operator: Any,
    inputs: Sequence[str],
    outputs: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> IRNode:
    meta = dict(metadata or {})
    meta.setdefault("operator_type", type(operator).__name__)
    wrapped = ensure_pytree_compatible(operator)
    meta.setdefault("is_pytree_safe", not isinstance(wrapped, StaticWrapper))
    return IRNode(
        id=node_id, operator=operator, inputs=tuple(inputs), outputs=tuple(outputs), metadata=meta
    )


__all__ = [
    "IRNode",
    "IRGraph",
    "ParallelismInfo",
    "GraphParallelismAnalysis",
    "node_from_callable",
]
