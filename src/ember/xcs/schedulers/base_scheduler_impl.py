"""Base implementation for scheduler strategies.

Provides abstract base classes and mixin patterns for scheduler implementations,
allowing for composition of ordering and execution strategies.
"""

import abc
import logging
from typing import Any, Dict, List, Optional, Set

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.schedulers.base_scheduler import BaseScheduler
from ember.xcs.utils.boundary import to_ember_model, to_dict

logger = logging.getLogger(__name__)


class OrderingStrategy(abc.ABC):
    """Strategy for determining execution order of graph nodes."""
    
    @abc.abstractmethod
    def get_execution_order(self, graph: XCSGraph) -> List[str]:
        """Determine the execution order for graph nodes.
        
        Args:
            graph: Graph to analyze
            
        Returns:
            List of node IDs in execution order
        """
        pass


class ExecutionStrategy(abc.ABC):
    """Strategy for executing graph nodes."""
    
    @abc.abstractmethod
    def execute_nodes(
        self,
        graph: XCSGraph,
        execution_order: List[str],
        inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute graph nodes in given order.
        
        Args:
            graph: Graph to execute
            execution_order: Node IDs in execution order
            inputs: Input values for the graph
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        pass
    
    @abc.abstractmethod
    def get_partial_results(self) -> Dict[str, Dict[str, Any]]:
        """Get partial results from execution.
        
        Returns:
            Dictionary mapping node IDs to their output results
        """
        pass


class TopologicalOrderingStrategy(OrderingStrategy):
    """Orders nodes topologically, with dependencies before dependents."""
    
    def get_execution_order(self, graph: XCSGraph) -> List[str]:
        """Get topological execution order for graph nodes.
        
        Args:
            graph: Graph to analyze
            
        Returns:
            List of node IDs in topological order
        """
        return graph.topological_sort()


class DepthOrderingStrategy(OrderingStrategy):
    """Orders nodes by depth in the dependency graph."""
    
    def get_execution_order(self, graph: XCSGraph) -> List[str]:
        """Get depth-based execution order for graph nodes.
        
        Assigns depths to nodes (distance from root) and orders them
        by increasing depth for layer-by-layer execution.
        
        Args:
            graph: Graph to analyze
            
        Returns:
            List of node IDs in depth-first order
        """
        # Compute node depths (distance from inputs)
        depths: Dict[str, int] = {}
        visited: Set[str] = set()
        
        # Find input nodes (no inbound edges)
        input_nodes = [node_id for node_id, node in graph.nodes.items() 
                      if not node.inbound_edges]
        
        # Assign depth 0 to input nodes
        for node_id in input_nodes:
            depths[node_id] = 0
        
        # BFS to compute depths
        queue = input_nodes.copy()
        while queue:
            node_id = queue.pop(0)
            visited.add(node_id)
            node_depth = depths[node_id]
            
            # Update depths of outbound nodes
            for outbound_id in graph.nodes[node_id].outbound_edges:
                if outbound_id not in depths or depths[outbound_id] < node_depth + 1:
                    depths[outbound_id] = node_depth + 1
                
                # Add to queue if all inbound nodes have been visited
                outbound_node = graph.nodes[outbound_id]
                if outbound_id not in visited and all(in_id in visited 
                                                    for in_id in outbound_node.inbound_edges):
                    queue.append(outbound_id)
        
        # Sort nodes by depth
        return sorted(graph.nodes.keys(), key=lambda node_id: depths.get(node_id, float('inf')))


class WaveOrderingStrategy(OrderingStrategy):
    """Orders nodes by execution waves for parallel processing."""
    
    def get_execution_order(self, graph: XCSGraph) -> List[str]:
        """Get wave-based execution order for graph nodes.
        
        Groups nodes into execution waves based on dependencies,
        where all nodes in a wave can be executed in parallel.
        
        Args:
            graph: Graph to analyze
            
        Returns:
            List of node IDs with wave annotations in metadata
        """
        # Track remaining dependencies for each node
        in_degree = {}
        for node_id, node in graph.nodes.items():
            in_degree[node_id] = len(node.inbound_edges)
        
        # Group nodes into waves for parallel execution
        waves = []
        wave_map = {}
        
        # Add nodes to waves based on dependency satisfaction
        while in_degree:
            # Find nodes with no remaining dependencies
            current_wave = [node_id for node_id, count in in_degree.items() if count == 0]
            if not current_wave:
                # Cyclic dependency detected
                raise ValueError("Graph contains a cycle")
            
            # Add current wave to waves list
            waves.append(current_wave)
            
            # Map nodes to their wave number
            wave_num = len(waves) - 1
            for node_id in current_wave:
                wave_map[node_id] = wave_num
            
            # Remove processed nodes and update dependencies
            for node_id in current_wave:
                for outbound_id in graph.nodes[node_id].outbound_edges:
                    in_degree[outbound_id] -= 1
                del in_degree[node_id]
        
        # Add wave metadata to graph for later use in execution
        graph.metadata["waves"] = waves
        graph.metadata["wave_map"] = wave_map
        
        # Return flattened list while preserving wave information
        return [node_id for wave in waves for node_id in wave]


class SequentialExecutionStrategy(ExecutionStrategy):
    """Executes nodes sequentially in specified order."""
    
    def __init__(self) -> None:
        """Initialize sequential execution strategy."""
        self._results: Dict[str, Dict[str, Any]] = {}
    
    def execute_nodes(
        self,
        graph: XCSGraph,
        execution_order: List[str],
        inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute graph nodes sequentially.
        
        Args:
            graph: Graph to execute
            execution_order: Node IDs in execution order
            inputs: Input values for the graph
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        self._results = {}  # Clear previous results
        
        # Initialize shared results with inputs
        shared_results = {}
        
        # Process nodes in order
        for node_id in execution_order:
            node = graph.nodes[node_id]
            
            # Prepare node inputs
            node_inputs = graph.prepare_node_inputs(node_id, self._results)
            # Add shared inputs for nodes with no inbound edges
            if not node.inbound_edges:
                node_inputs.update(inputs)
            
            # Ensure inputs have the right type for the target node
            
            # Get expected input type from node's operator specification if available
            target_operator = node.operator
            input_model = None
            if hasattr(target_operator, "specification") and hasattr(
                target_operator.specification, "input_model"
            ):
                input_model = target_operator.specification.input_model
            
            # Execute the operator with boundary crossing
            try:
                # Always cross boundary when entering and exiting operator
                if input_model:
                    # ENTER: Convert dict to model when crossing into operator
                    logger.debug(f"Node {node_id}: Boundary crossing: dict → {input_model.__name__}")
                    typed_inputs = to_ember_model(node_inputs, input_model)
                    
                    # Execute in operator domain (with proper types)
                    result = node.operator(inputs=typed_inputs)
                    
                    # EXIT: Convert result back to dict when crossing back to execution engine
                    logger.debug(f"Node {node_id}: Boundary crossing: {type(result).__name__} → dict")
                    self._results[node_id] = to_dict(result)
                else:
                    # No model type specified - simpler execution path
                    result = node.operator(inputs=node_inputs)
                    # Still ensure dict format on return for consistency
                    self._results[node_id] = to_dict(result)
            except TypeError as e:
                # Specific handling for boundary crossing errors
                logger.error(f"Boundary crossing error at node {node_id}: {e}")
                raise RuntimeError(f"Type error at system boundary for node {node_id}: {e}") from e
            except Exception as e:
                logger.error(f"Error executing node {node_id}: {e}")
                # Store partial successful results
                self._results[node_id] = {"error": str(e)}
                # Propagate error
                raise RuntimeError(f"Error executing node {node_id}: {e}") from e
        
        return self._results
    
    def get_partial_results(self) -> Dict[str, Dict[str, Any]]:
        """Get partial results from execution.
        
        Returns:
            Dictionary mapping node IDs to their output results
        """
        return self._results


class ParallelExecutionStrategy(ExecutionStrategy):
    """Executes nodes in parallel where possible."""
    
    def __init__(self, max_workers: Optional[int] = None) -> None:
        """Initialize parallel execution strategy.
        
        Args:
            max_workers: Maximum number of worker threads (None uses CPU count)
        """
        self.max_workers = max_workers
        self._results: Dict[str, Dict[str, Any]] = {}
        
    def _execute_and_convert(self, operator: Any, inputs: Any, boundary_debug_id: str = "") -> Dict[str, Any]:
        """Execute an operator and convert its result to a dictionary.
        
        Helper method for boundary crossing in parallel execution.
        
        Args:
            operator: Operator to execute
            inputs: Properly typed inputs for the operator
            boundary_debug_id: Optional identifier for debugging boundary crossing
            
        Returns:
            Dictionary representation of the operator result
        """
        # Execute the operator with its expected input types
        result = operator(inputs=inputs)
        
        # EXIT boundary: Always ensure dictionary format when returning to execution engine
        if not isinstance(result, dict):
            logger.debug(f"Boundary crossing EXIT {boundary_debug_id}: {type(result).__name__} → dict")
            return to_dict(result)
        return result
    
    def execute_nodes(
        self,
        graph: XCSGraph,
        execution_order: List[str],
        inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute graph nodes with parallel processing where possible.
        
        Args:
            graph: Graph to execute
            execution_order: Node IDs in execution order
            inputs: Input values for the graph
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        
        self._results = {}  # Clear previous results
        
        # Check if wave information is available for parallel execution
        waves = graph.metadata.get("waves")
        if waves:
            # Execute by waves using wave information
            return self._execute_by_waves(graph, waves, inputs)
            
        # Fallback: determine parallelizable nodes using dependencies
        return self._execute_with_dependencies(graph, execution_order, inputs)
    
    def _execute_by_waves(self, 
                         graph: XCSGraph, 
                         waves: List[List[str]], 
                         inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Execute nodes in waves, with each wave executed in parallel.
        
        Args:
            graph: Graph to execute
            waves: List of waves, where each wave is a list of node IDs
            inputs: Input values for the graph
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        
        # Initialize results
        self._results = {}
        
        # Calculate appropriate worker count
        workers = self.max_workers
        if workers is None:
            import multiprocessing
            workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Process each wave in order
        for wave in waves:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Prepare futures for this wave
                futures = {}
                for node_id in wave:
                    node = graph.nodes[node_id]
                    # Prepare inputs for this node
                    node_inputs = graph.prepare_node_inputs(node_id, self._results)
                    # Add shared inputs for nodes with no inbound edges
                    if not node.inbound_edges:
                        node_inputs.update(inputs)
                    
                    # Ensure inputs have the right type for the target node
                    
                    # Get expected input type from node's operator specification if available
                    target_operator = node.operator
                    input_model = None
                    if hasattr(target_operator, "specification") and hasattr(
                        target_operator.specification, "input_model"
                    ):
                        input_model = target_operator.specification.input_model
                    
                    # Always apply consistent boundary crossing
                    try:
                        if input_model:
                            # ENTER boundary: Convert dict to model when crossing into operator
                            logger.debug(f"Node {node_id}: Boundary crossing ENTER: dict → {input_model.__name__}")
                            typed_inputs = to_ember_model(node_inputs, input_model)
                            # Submit the task with properly typed inputs through conversion helper
                            futures[executor.submit(
                                self._execute_and_convert, 
                                node.operator, 
                                typed_inputs,
                                f"node={node_id}"
                            )] = node_id
                        else:
                            # No model type specified - simpler execution path, but still ensure dict return
                            futures[executor.submit(
                                self._execute_and_convert,
                                node.operator,
                                node_inputs,
                                f"node={node_id}"
                            )] = node_id
                    except TypeError as e:
                        # Handle conversion error
                        logger.error(f"Boundary crossing error at node {node_id}: {e}")
                        raise RuntimeError(f"Type error at system boundary for node {node_id}: {e}") from e
                
                # Wait for all tasks in this wave to complete
                for future in concurrent.futures.as_completed(futures):
                    node_id = futures[future]
                    try:
                        result = future.result()
                        self._results[node_id] = result
                    except Exception as e:
                        logger.error(f"Error executing node {node_id}: {e}")
                        # Store partial successful results
                        self._results[node_id] = {"error": str(e)}
                        # Propagate error
                        raise RuntimeError(f"Error executing node {node_id}: {e}") from e
        
        return self._results
    
    def _execute_with_dependencies(
        self,
        graph: XCSGraph,
        execution_order: List[str],
        inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute nodes using dependency tracking for parallelism.
        
        Args:
            graph: Graph to execute
            execution_order: Node IDs in execution order
            inputs: Input values for the graph
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        # Initialize tracking structures
        self._results = {}
        pending_nodes = set(execution_order)
        completed_nodes = set()
        node_lock = threading.Lock()
        
        # Calculate appropriate worker count
        workers = self.max_workers
        if workers is None:
            import multiprocessing
            workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Execute nodes with dependency tracking
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit initial nodes (those with no dependencies)
            futures = {}
            initial_nodes = [node_id for node_id in execution_order 
                          if not graph.nodes[node_id].inbound_edges]
            
            for node_id in initial_nodes:
                # Prepare inputs
                node_inputs = graph.prepare_node_inputs(node_id, self._results)
                node_inputs.update(inputs)  # Add shared inputs
                
                # Submit the task
                futures[executor.submit(self._execute_node, graph, node_id, node_inputs, 
                                      completed_nodes, pending_nodes, node_lock, executor, 
                                      futures, inputs)] = node_id
            
            # Wait for all tasks to complete
            try:
                concurrent.futures.wait(futures.keys())
            except Exception as e:
                logger.error(f"Error during parallel execution: {e}")
                raise
            
        return self._results
    
    def _execute_node(
        self,
        graph: XCSGraph,
        node_id: str,
        node_inputs: Dict[str, Any],
        completed_nodes: Set[str],
        pending_nodes: Set[str],
        node_lock: Any,
        executor: Any,
        futures: Dict[Any, str],
        global_inputs: Dict[str, Any]
    ) -> None:
        """Execute a single node and schedule its dependents.
        
        Args:
            graph: The graph being executed
            node_id: ID of the node to execute
            node_inputs: Inputs for the node
            completed_nodes: Set of completed node IDs
            pending_nodes: Set of pending node IDs
            node_lock: Lock for synchronizing access to shared state
            executor: ThreadPoolExecutor for submitting new tasks
            futures: Dictionary mapping futures to node IDs
            global_inputs: Global inputs to the graph
        """
        node = graph.nodes[node_id]
        
        try:
            # Execute the node - node_inputs are already properly typed
            # at this point due to the boundary crossing in the caller
            result = node.operator(inputs=node_inputs)
            
            # EXIT: Convert result back to dictionary when crossing back to XCS
            logger.debug(f"Node {node_id}: Boundary crossing EXIT: {type(result).__name__} → dict")
            dict_result = to_dict(result)
            
            # Update shared state
            with node_lock:
                self._results[node_id] = dict_result
                completed_nodes.add(node_id)
                pending_nodes.remove(node_id)
                
                # Check for nodes that can now be executed
                for outbound_id in node.outbound_edges:
                    if outbound_id in pending_nodes:
                        # Check if all dependencies are satisfied
                        outbound_node = graph.nodes[outbound_id]
                        if all(dep_id in completed_nodes for dep_id in outbound_node.inbound_edges):
                            # Prepare inputs for this node
                            outbound_inputs = graph.prepare_node_inputs(outbound_id, self._results)
                            
                            # Ensure inputs have the right type for the target node
                            
                            # Get expected input type from node's operator specification if available
                            target_operator = outbound_node.operator
                            input_model = None
                            if hasattr(target_operator, "specification") and hasattr(
                                target_operator.specification, "input_model"
                            ):
                                input_model = target_operator.specification.input_model
                            
                            # Always apply consistent boundary crossing at the entry point
                            try:
                                if input_model:
                                    # ENTER boundary: Convert dict to model when crossing into operator
                                    logger.debug(f"Node {outbound_id}: Boundary crossing ENTER: dict → {input_model.__name__}")
                                    typed_inputs = to_ember_model(outbound_inputs, input_model)
                                    # Submit the task with properly typed inputs
                                    futures[executor.submit(self._execute_node, graph, outbound_id, 
                                                          typed_inputs, completed_nodes, pending_nodes, 
                                                          node_lock, executor, futures, global_inputs)] = outbound_id
                                else:
                                    # No model specified, use as-is but ensure dict return
                                    futures[executor.submit(self._execute_node, graph, outbound_id, 
                                                          outbound_inputs, completed_nodes, pending_nodes, 
                                                          node_lock, executor, futures, global_inputs)] = outbound_id
                            except TypeError as e:
                                # Handle conversion error
                                logger.error(f"Boundary crossing error at node {outbound_id}: {e}")
                                with node_lock:
                                    self._results[outbound_id] = {"error": f"Type error at system boundary: {e}"}
                                raise
        
        except Exception as e:
            logger.error(f"Error executing node {node_id}: {e}")
            # Store error information as a proper dict (will be converted to correct model type by prepare_node_inputs)
            with node_lock:
                self._results[node_id] = {"error": str(e)}
            raise
    
    def get_partial_results(self) -> Dict[str, Dict[str, Any]]:
        """Get partial results from execution.
        
        Returns:
            Dictionary mapping node IDs to their output results
        """
        return self._results


class NoopExecutionStrategy(ExecutionStrategy):
    """Non-executing strategy for testing and validation."""
    
    def __init__(self) -> None:
        """Initialize no-op execution strategy."""
        self._results: Dict[str, Dict[str, Any]] = {}
    
    def execute_nodes(
        self,
        graph: XCSGraph,
        execution_order: List[str],
        inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Return inputs without executing nodes (for simulation).
        
        Args:
            graph: Graph to not execute
            execution_order: Node IDs in execution order
            inputs: Input values for the graph
            
        Returns:
            Dictionary mapping node IDs to placeholder results
        """
        self._results = {}
        
        # Create placeholder results for each node
        for node_id in execution_order:
            self._results[node_id] = {"_simulated": True}
        
        return self._results
    
    def get_partial_results(self) -> Dict[str, Dict[str, Any]]:
        """Get partial results from execution.
        
        Returns:
            Dictionary mapping node IDs to their output results
        """
        return self._results


class BaseSchedulerImpl(BaseScheduler):
    """Base implementation for all schedulers using strategy pattern.
    
    Combines ordering and execution strategies to implement the full
    scheduler interface. Concrete schedulers can extend this class and
    provide specific strategy combinations.
    """
    
    def __init__(
        self,
        ordering_strategy: OrderingStrategy,
        execution_strategy: ExecutionStrategy
    ) -> None:
        """Initialize with specific strategies.
        
        Args:
            ordering_strategy: Strategy for determining execution order
            execution_strategy: Strategy for executing nodes
        """
        self._ordering_strategy = ordering_strategy
        self._execution_strategy = execution_strategy
        self._execution_order: List[str] = []
    
    def prepare(self, graph: XCSGraph) -> None:
        """Prepare the scheduler for graph execution.
        
        Args:
            graph: The graph to be executed
        """
        # Determine execution order using ordering strategy
        self._execution_order = self._ordering_strategy.get_execution_order(graph)
    
    def execute(self, graph: XCSGraph, inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Execute all nodes in the graph using execution strategy.
        
        Args:
            graph: The graph to execute
            inputs: Initial inputs to the graph
            
        Returns:
            Dictionary mapping node IDs to their output results
        """
        return self._execution_strategy.execute_nodes(graph, self._execution_order, inputs)
    
    def get_partial_results(self) -> Dict[str, Dict[str, Any]]:
        """Get partial results from execution that may have been interrupted.
        
        Returns:
            Dictionary mapping node IDs to their output results
        """
        return self._execution_strategy.get_partial_results()