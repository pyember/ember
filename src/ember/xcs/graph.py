"""What a jit call executed: its model calls and the reads that ordered them.

A `Graph` is a record of one run, built from measurements taken during the
run. Nothing in it is estimated, and it is never executed in place of the
function that produced it.
"""

import dataclasses

__all__ = ["Graph", "Node"]


@dataclasses.dataclass(frozen=True, slots=True)
class Node:
    """One model call in a recorded run.

    Attributes:
        id: Position in dispatch order, from 0.
        label: The label the call was dispatched with, usually the model id.
        deps: Ids of the calls whose results the dispatching code had read
            before it made this call, in increasing order.
        start_s: Seconds from the start of the jit call to the dispatch.
        end_s: Seconds from the start of the jit call to the result.
    """

    id: int
    label: str
    deps: tuple[int, ...]
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Seconds from dispatch to result."""
        return self.end_s - self.start_s


@dataclasses.dataclass(frozen=True, slots=True)
class Graph:
    """The model calls of one jit call, in dispatch order.

    An edge ``(a, b)`` means that the code which dispatched ``b`` had read the
    result of ``a`` first, directly or through results that depended on it.

    Attributes:
        nodes: The calls; ``nodes[i].id == i``.
        wall_s: Seconds the jit call took.

    Raises:
        ValueError: On construction, if ids are not positions, a dependency
            does not precede its dependent, or a node ends before it starts.
    """

    nodes: tuple[Node, ...]
    wall_s: float

    def __post_init__(self) -> None:
        for position, node in enumerate(self.nodes):
            if node.id != position:
                raise ValueError(f"node at position {position} has id {node.id}")
            if list(node.deps) != sorted(set(node.deps)):
                raise ValueError(f"node {node.id} deps must be increasing and unique")
            if node.deps and (node.deps[0] < 0 or node.deps[-1] >= node.id):
                raise ValueError(f"node {node.id} depends on a node that does not precede it")
            if node.end_s < node.start_s:
                raise ValueError(f"node {node.id} ends before it starts")

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Every ``(dependency, dependent)`` pair."""
        return tuple((dep, node.id) for node in self.nodes for dep in node.deps)

    def remaining_s(self) -> tuple[float, ...]:
        """For each node, its duration plus the longest chain of durations after it."""
        successors = self._successors()
        remaining = [0.0] * len(self.nodes)
        for node in reversed(self.nodes):
            after = max((remaining[s] for s in successors[node.id]), default=0.0)
            remaining[node.id] = node.duration_s + after
        return tuple(remaining)

    @property
    def critical_path(self) -> tuple[int, ...]:
        """Ids along the chain of dependent calls with the largest total duration."""
        if not self.nodes:
            return ()
        remaining = self.remaining_s()
        successors = self._successors()
        path = [max(range(len(self.nodes)), key=lambda i: (remaining[i], -i))]
        while successors[path[-1]]:
            path.append(max(successors[path[-1]], key=lambda i: (remaining[i], -i)))
        return tuple(path)

    def _successors(self) -> list[list[int]]:
        successors: list[list[int]] = [[] for _ in self.nodes]
        for dep, dependent in self.edges:
            successors[dep].append(dependent)
        return successors
