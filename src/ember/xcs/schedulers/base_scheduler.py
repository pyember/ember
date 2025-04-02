"""
Base scheduler protocol and implementation for XCS.

Defines the core interface for graph execution schedulers and provides base
implementations that other schedulers can extend.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, Set, Union

from ember.xcs.common.plans import XCSPlan
from ember.xcs.graph.xcs_graph import XCSGraph


class BaseScheduler(Protocol):
    """Protocol defining the interface for graph execution schedulers.
    
    All schedulers must implement this protocol to ensure consistent behavior
    and interoperability with the execution engine.
    """
    
    def prepare(self, graph: XCSGraph) -> None:
        """Prepare the scheduler for execution.
        
        This method is called before execution begins to initialize the
        scheduler with the graph structure.
        
        Args:
            graph: The graph to prepare for execution
        """
        ...
        
    def execute(self, graph: XCSGraph, inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Execute a graph with the given inputs.
        
        This is the main method for running a graph through the scheduler.
        
        Args:
            graph: The graph to execute
            inputs: Input values for the graph
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        ...
        
    def get_partial_results(self) -> Dict[str, Dict[str, Any]]:
        """Get partial results from an incomplete execution.
        
        This is used when execution is stopped prematurely (e.g., due to
        an error or timeout) to retrieve partial results.
        
        Returns:
            Dictionary mapping node IDs to their output results
        """
        ...


class UnifiedSchedulerBase:
    """Base implementation of scheduler with unified execution capabilities.
    
    This class serves as a foundation for all scheduler implementations,
    providing common execution capabilities through the execution coordinator.
    It enables schedulers to use both thread pool and async execution engines
    depending on the workload characteristics.
    
    This is separate from the BaseScheduler protocol to maintain backward
    compatibility with existing scheduler implementations while offering
    enhanced execution capabilities for new scheduler implementations.
    """
    
    def __init__(
        self, 
        max_workers: Optional[int] = None, 
        execution_engine: str = "auto",
        timeout_seconds: Optional[float] = None,
        error_handling: str = "fail_fast"
    ) -> None:
        """Initialize scheduler with execution parameters.
        
        Args:
            max_workers: Maximum concurrent operations
            execution_engine: Engine selection - one of:
                "auto" - automatically select based on workload
                "async" - use async/await for IO-bound operations
                "threaded" - use thread pool for CPU-bound operations
            timeout_seconds: Maximum execution time in seconds
            error_handling: Error handling strategy - one of:
                "fail_fast" - stop execution on first error
                "continue" - continue execution after errors when possible
        """
        self.max_workers = max_workers
        self.execution_engine = execution_engine
        self.timeout_seconds = timeout_seconds
        self.error_handling = error_handling
        self.logger = logging.getLogger(__name__)
        self._results: Dict[str, Dict[str, Any]] = {}
        
    def _get_execution_order(self, graph: XCSGraph) -> List[List[str]]:
        """Get the execution order for nodes in the graph.
        
        Must be implemented by concrete scheduler classes.
        
        Args:
            graph: Graph to analyze
            
        Returns:
            List of waves, where each wave is a list of node IDs
        """
        raise NotImplementedError("Subclasses must implement _get_execution_order")
    
    def _prepare_node_input(
        self, 
        node_id: str, 
        node: Any, 
        results: Dict[str, Dict[str, Any]],
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for a node based on graph connections.
        
        Args:
            node_id: ID of node being prepared
            node: Node being prepared
            results: Results collected so far
            inputs: Original graph inputs
            
        Returns:
            Properly prepared inputs for the node
        """
        # Default implementation uses graph's prepare_node_inputs
        # Can be overridden by subclasses for custom behavior
        node_inputs = {}
        
        # If this is a source node with no dependencies, use graph inputs
        if not node.inbound_edges:
            node_inputs.update(inputs)
            return node_inputs
            
        # Otherwise, collect results from dependencies
        for source_id in node.inbound_edges:
            if source_id in results:
                # For now, just shallow merge results
                # More sophisticated implementations can use field_mappings
                node_inputs.update(results[source_id])
                
        return node_inputs
    
    def run_plan(
        self, *, plan: XCSPlan, inputs: Dict[str, Any], graph: XCSGraph
    ) -> Dict[str, Any]:
        """Execute plan with adaptive execution engine.
        
        Args:
            plan: Execution plan to run
            inputs: Input values for the plan
            graph: Computation graph
            
        Returns:
            Results from plan execution
        """
        # Import here to avoid circular imports
        from ember.xcs.utils.executor import ExecutionCoordinator
        
        # Create execution coordinator with optimal configuration
        coordinator = ExecutionCoordinator(
            max_workers=self.max_workers,
            timeout=self.timeout_seconds,
            error_handling=self.error_handling,
            execution_engine=self.execution_engine
        )
        
        try:
            results = {}
            # Process execution waves in order
            for wave in self._get_execution_order(graph):
                # Prepare wave nodes with inputs
                wave_nodes = [
                    {
                        "id": node_id,
                        "operator": graph.nodes[node_id].operator,
                        "inputs": self._prepare_node_input(
                            node_id, graph.nodes[node_id], results, inputs
                        )
                    }
                    for node_id in wave
                ]
                
                # Skip empty waves
                if not wave_nodes:
                    continue
                    
                # Execute wave with optimal engine selection
                wave_results = coordinator.map(
                    lambda **kwargs: kwargs["operator"](inputs=kwargs["inputs"]),
                    wave_nodes
                )
                
                # Store results by node ID
                for i, node_dict in enumerate(wave_nodes):
                    if i < len(wave_results):
                        node_id = node_dict["id"]
                        results[node_id] = wave_results[i]
                        
            # Store the final results for potential retrieval via get_partial_results
            self._results = results
            return results
        finally:
            coordinator.close()
            
    def get_partial_results(self) -> Dict[str, Dict[str, Any]]:
        """Get partial results from execution that may have been interrupted.
        
        Returns:
            Dictionary mapping node IDs to their output results
        """
        return self._results