"""
Dependency analysis for XCS computation graphs.

Provides unified dependency tracking and analysis for all graph operations,
supporting topological sorting, transitive closure calculation, and execution
wave computation for parallel scheduling.
"""

from typing import Dict, List, Set

from ember.xcs.graph.xcs_graph import XCSGraph


class DependencyAnalyzer:
    """Unified dependency analyzer for XCS graphs.
    
    Analyzes node dependencies and constructs dependency graphs for
    scheduling and optimization purposes.
    """
    
    def analyze(self, graph: XCSGraph) -> Dict[str, Set[str]]:
        """Analyze all dependencies in a graph.
        
        Computes the complete dependency relationship between all nodes,
        including transitive dependencies.
        
        Args:
            graph: The graph to analyze
            
        Returns:
            Dictionary mapping each node to its complete set of dependencies
            (direct and transitive)
        """
        direct_deps = self.build_dependency_graph(graph)
        return self.compute_transitive_closure(direct_deps)
    
    def build_dependency_graph(self, graph: XCSGraph) -> Dict[str, Set[str]]:
        """Build a direct dependency graph.
        
        Constructs a mapping of each node to its direct dependencies.
        
        Args:
            graph: The graph to analyze
            
        Returns:
            Dictionary mapping each node to its direct dependencies
        """
        direct_deps: Dict[str, Set[str]] = {}
        
        # For each node, determine its direct dependencies
        for node_id, node in graph.nodes.items():
            direct_deps[node_id] = set(node.inbound_edges)
            
        return direct_deps
    
    def compute_transitive_closure(self, direct_deps: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Compute the transitive closure of dependencies.
        
        Determines all direct and indirect dependencies for each node using
        a fixed-point algorithm.
        
        Args:
            direct_deps: Direct dependency mapping from node IDs to sets of
                         direct dependency node IDs
            
        Returns:
            Dictionary mapping each node to all dependencies
            (direct and transitive)
        """
        # Initialize with direct dependencies
        all_deps: Dict[str, Set[str]] = {
            node: set(deps) for node, deps in direct_deps.items()
        }
        
        # Add empty sets for nodes that appear as dependencies but don't have dependencies
        all_deps_nodes = set(all_deps.keys())
        all_dep_targets = {dep for deps in all_deps.values() for dep in deps}
        for node in all_dep_targets:
            if node not in all_deps_nodes:
                all_deps[node] = set()
        
        # Repeatedly update dependency sets until no more changes
        changed = True
        while changed:
            changed = False
            for node, deps in all_deps.items():
                new_deps = deps.copy()
                
                # Add transitive dependencies
                for dep in deps:
                    if dep in all_deps:
                        new_deps.update(all_deps[dep])
                
                # Check if the set changed
                if new_deps != deps:
                    all_deps[node] = new_deps
                    changed = True
        
        return all_deps
    
    def topological_sort(self, graph: XCSGraph) -> List[str]:
        """Perform topological sort on the graph nodes.
        
        Creates a linear ordering of nodes such that for every directed edge
        (A, B), node A comes before node B in the ordering.
        
        Args:
            graph: The graph to sort
            
        Returns:
            List of node IDs in topological order
            
        Raises:
            ValueError: If the graph contains cycles
        """
        # Use the graph's built-in topological_sort if available
        if hasattr(graph, "topological_sort") and callable(graph.topological_sort):
            return graph.topological_sort()
            
        # Fallback implementation
        direct_deps = self.build_dependency_graph(graph)
        
        # Build dependency count map
        dependency_count: Dict[str, int] = {
            node: len(deps) for node, deps in direct_deps.items()
        }
        
        # Build reverse dependency map
        reverse_deps: Dict[str, Set[str]] = {node: set() for node in direct_deps}
        for node, deps in direct_deps.items():
            for dep in deps:
                if dep in reverse_deps:
                    reverse_deps[dep].add(node)
                else:
                    reverse_deps[dep] = {node}
        
        # Start with nodes that have no dependencies
        sorted_nodes: List[str] = []
        no_deps = [node for node, count in dependency_count.items() if count == 0]
        
        # Process nodes in topological order
        while no_deps:
            current = no_deps.pop(0)
            sorted_nodes.append(current)
            
            # Update dependency counts for nodes that depend on current
            for dependent in reverse_deps.get(current, set()):
                dependency_count[dependent] -= 1
                if dependency_count[dependent] == 0:
                    no_deps.append(dependent)
        
        # Check for cycles
        if len(sorted_nodes) != len(direct_deps):
            raise ValueError("Graph contains cycles and cannot be topologically sorted")
        
        return sorted_nodes
    
    def compute_execution_waves(self, graph: XCSGraph) -> List[List[str]]:
        """Compute execution waves for parallel scheduling.
        
        Groups nodes into waves where nodes in each wave have no dependencies
        on each other and can be executed in parallel.
        
        Args:
            graph: The graph to analyze
            
        Returns:
            List of waves, each containing node IDs that can execute in parallel
            
        Raises:
            ValueError: If the graph contains cycles
        """
        # Get direct dependencies
        direct_deps = self.build_dependency_graph(graph)
        
        # Group nodes into waves
        waves: List[List[str]] = []
        remaining_nodes = set(direct_deps.keys())
        
        while remaining_nodes:
            # Find nodes with no dependencies in the remaining set
            current_wave = [
                node for node in remaining_nodes 
                if all(dep not in remaining_nodes for dep in direct_deps.get(node, set()))
            ]
            
            # Check for cycles
            if not current_wave:
                raise ValueError("Graph contains cycles and cannot be executed in waves")
            
            # Add the current wave
            waves.append(current_wave)
            
            # Remove processed nodes
            remaining_nodes -= set(current_wave)
        
        return waves