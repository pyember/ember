"""Parallelism analysis for XCS IR graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Set

from ember.xcs.compiler.graph import (
    GraphParallelismAnalysis,
    IRGraph,
    IRNode,
    ParallelismInfo,
)

# Maximum steps for bounded path search. Prevents O(V²) blowup on large graphs
# while still detecting dependencies in typical graphs (<100 nodes).
_MAX_PATH_SEARCH_STEPS = 200


@dataclass(slots=True)
class _DepthInfo:
    node_id: str
    depth: int


class ParallelismAnalyzer:
    """Discover vectorization and parallel execution opportunities.

    Uses lazy independence checking instead of computing full transitive closure.
    This gives O(k² * E) where k is the number of candidate parallel nodes at each
    depth level, instead of O(V²) for full reachability computation.
    """

    def analyze(self, graph: IRGraph) -> GraphParallelismAnalysis:
        node_info = {
            node_id: self._analyze_node(node, graph) for node_id, node in graph.nodes.items()
        }

        parallel_groups = self._parallel_groups(graph)
        vectorizable = self._vectorizable_chains(graph, node_info)
        bottlenecks = self._bottlenecks(graph, node_info, parallel_groups)
        speedup = self._estimate_speedup(node_info, parallel_groups, vectorizable)
        return GraphParallelismAnalysis(
            node_info=node_info,
            parallel_groups=parallel_groups,
            vectorizable_chains=vectorizable,
            estimated_speedup=speedup,
            bottlenecks=bottlenecks,
        )

    def _analyze_node(self, node: IRNode, graph: IRGraph) -> ParallelismInfo:
        metadata = node.metadata
        can_parallelize = bool(metadata.get("is_comprehension"))
        can_vmap = metadata.get("is_pytree_safe", False)
        can_pmap = bool(metadata.get("distributed_hint"))
        estimated = 1.0
        if can_parallelize:
            dependents = len(graph.get_dependents(node.id))
            estimated = max(estimated, float(max(1, dependents)))
        if can_vmap:
            estimated = max(estimated, 2.0)
        if can_pmap:
            estimated = max(estimated, 4.0)
        is_pure = bool(metadata.get("is_pure", True))
        return ParallelismInfo(
            can_vmap=can_vmap,
            can_pmap=can_pmap,
            can_parallelize=can_parallelize,
            estimated_speedup=estimated,
            is_pure=is_pure,
        )

    def _parallel_groups(self, graph: IRGraph) -> List[Set[str]]:
        """Discover groups of nodes that can execute in parallel.

        Uses depth bucketing to find candidates, then lazy pairwise independence
        checking to confirm they have no data dependencies. This avoids computing
        the full O(V²) transitive closure.
        """
        depth_map = self._depths(graph)
        buckets: Dict[int, List[str]] = {}
        for info in depth_map:
            buckets.setdefault(info.depth, []).append(info.node_id)

        groups: List[Set[str]] = []
        for node_ids in buckets.values():
            if len(node_ids) <= 1:
                continue
            independent = self._find_independent_subset(graph, node_ids)
            if len(independent) > 1:
                groups.append(independent)
        return groups

    def _find_independent_subset(
        self, graph: IRGraph, candidates: List[str]
    ) -> Set[str]:
        """Find a maximal subset of candidates that are mutually independent.

        Greedily builds an independent set by checking each candidate against
        already-selected nodes. O(k² * E) where k = len(candidates).
        """
        independent: Set[str] = set()
        for node_id in candidates:
            is_independent = all(
                not self._has_path_bounded(graph, node_id, other)
                and not self._has_path_bounded(graph, other, node_id)
                for other in independent
            )
            if is_independent:
                independent.add(node_id)
        return independent

    def _has_path_bounded(
        self, graph: IRGraph, source: str, target: str
    ) -> bool:
        """Check if a path exists from source to target with bounded work.

        Returns True if:
        - A path is found, OR
        - The search budget is exceeded (conservative: assume dependent)

        This prevents O(V) traversal per pair while still catching most
        dependencies in reasonably-sized graphs.
        """
        if source == target:
            return True

        visited: Set[str] = set()
        stack = [source]
        steps = 0

        while stack and steps < _MAX_PATH_SEARCH_STEPS:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for dep in graph.get_dependents(current):
                if dep == target:
                    return True
                stack.append(dep)
            steps += 1

        # If we exceeded budget, conservatively assume dependent (no parallelism).
        # This is safe - we might miss some parallelization opportunities in very
        # large graphs, but we won't incorrectly parallelize dependent operations.
        return steps >= _MAX_PATH_SEARCH_STEPS

    def _vectorizable_chains(
        self, graph: IRGraph, info: Mapping[str, ParallelismInfo]
    ) -> List[List[str]]:
        chains: List[List[str]] = []
        visited: Set[str] = set()
        for node_id in graph.topological_order():
            if node_id in visited or not info[node_id].can_vmap:
                continue
            chain = [node_id]
            visited.add(node_id)
            current = node_id
            while True:
                dependents = graph.get_dependents(current)
                if len(dependents) != 1:
                    break
                next_node = dependents[0]
                if next_node in visited or not info[next_node].can_vmap:
                    break
                chain.append(next_node)
                visited.add(next_node)
                current = next_node
            if len(chain) > 1:
                chains.append(chain)
        return chains

    def _bottlenecks(
        self,
        graph: IRGraph,
        info: Mapping[str, ParallelismInfo],
        groups: Sequence[Set[str]],
    ) -> List[str]:
        bottlenecks: List[str] = []
        group_nodes = {nid for group in groups for nid in group}
        for node_id, _node in graph.nodes.items():
            deps = graph.get_dependencies(node_id)
            if len(deps) > 1 and any(dep in group_nodes for dep in deps):
                bottlenecks.append(node_id)
            if not info[node_id].can_parallelize and len(graph.get_dependents(node_id)) > 1:
                bottlenecks.append(node_id)
        return bottlenecks

    def _estimate_speedup(
        self,
        info: Mapping[str, ParallelismInfo],
        groups: Sequence[Set[str]],
        chains: Sequence[Sequence[str]],
    ) -> float:
        speedup = 1.0
        for group in groups:
            speedup = max(speedup, float(len(group)))
        for chain in chains:
            speedup = max(speedup, min(len(chain) * 2.0, 8.0))
        for _node_id, node_info in info.items():
            speedup = max(speedup, node_info.estimated_speedup)
        return min(speedup, 10.0)

    def _depths(self, graph: IRGraph) -> List[_DepthInfo]:
        depths: Dict[str, int] = {}
        order = graph.topological_order()
        for node_id in order:
            deps = graph.get_dependencies(node_id)
            resolved_deps = [dep for dep in deps if dep in depths]
            if not resolved_deps:
                depths[node_id] = 0
            else:
                depths[node_id] = 1 + max(depths[dep] for dep in resolved_deps)
        return [_DepthInfo(node_id, depth) for node_id, depth in depths.items()]


__all__ = ["ParallelismAnalyzer"]
