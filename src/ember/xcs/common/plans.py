"""Execution planning structures for XCS.

Defines data structures for representing execution plans and tasks in the
unified execution engine. These structures are crucial for translating
computation graphs into executable tasks.
"""

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Set


@dataclasses.dataclass
class XCSTask:
    """A single unit of execution within a plan.
    
    Represents a task to be executed by the scheduler, with its inputs,
    dependencies, and associated function/operator.
    
    Attributes:
        node_id: Unique identifier for this task
        operator: Function or operator to execute
        inputs: Input values for the operator
        dependencies: Task IDs that must complete before this one
        is_input_node: Whether this is an input node in the graph
        is_output_node: Whether this is an output node in the graph
    """
    
    node_id: str
    operator: Optional[Callable] = None
    inputs: Optional[Dict[str, Any]] = None
    dependencies: Set[str] = dataclasses.field(default_factory=set)
    is_input_node: bool = False
    is_output_node: bool = False
    
    def mark_as_input(self) -> None:
        """Mark this task as an input node."""
        self.is_input_node = True
        
    def mark_as_output(self) -> None:
        """Mark this task as an output node."""
        self.is_output_node = True
        
    def add_dependency(self, node_id: str) -> None:
        """Add a dependency to this task.
        
        Args:
            node_id: ID of the task that must complete before this one
        """
        self.dependencies.add(node_id)


@dataclasses.dataclass
class XCSPlan:
    """An execution plan built from a computation graph.
    
    Represents the full execution plan to be run by a scheduler, containing
    all tasks, their dependencies, and global input/output mappings.
    
    Attributes:
        tasks: Dictionary mapping node IDs to their task definitions
        input_nodes: Set of input node IDs
        output_nodes: Set of output node IDs
        global_input_mapping: Mapping from global inputs to specific node inputs
        global_output_mapping: Mapping from node outputs to global outputs
    """
    
    tasks: Dict[str, XCSTask] = dataclasses.field(default_factory=dict)
    input_nodes: Set[str] = dataclasses.field(default_factory=set)
    output_nodes: Set[str] = dataclasses.field(default_factory=set)
    global_input_mapping: Dict[str, Dict[str, str]] = dataclasses.field(default_factory=dict)
    global_output_mapping: Dict[str, str] = dataclasses.field(default_factory=dict)
    
    def add_task(self, task: XCSTask) -> None:
        """Add a task to the execution plan.
        
        Args:
            task: Task to add
        """
        self.tasks[task.node_id] = task
        if task.is_input_node:
            self.input_nodes.add(task.node_id)
        if task.is_output_node:
            self.output_nodes.add(task.node_id)
            
    def get_execution_order(self) -> List[str]:
        """Calculate a valid execution order for tasks.
        
        Performs a topological sort to determine a valid order for executing
        tasks respecting dependencies.
        
        Returns:
            List of node IDs in a valid execution order
        """
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node_id: str) -> None:
            """Recursive visit function for topological sort."""
            if node_id in visited:
                return
            if node_id in temp_visited:
                raise ValueError(f"Cycle detected in execution graph at node {node_id}")
                
            temp_visited.add(node_id)
            
            # Visit dependencies first
            for dep_id in self.tasks[node_id].dependencies:
                visit(dep_id)
                
            temp_visited.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
            
        # Visit all nodes
        for node_id in self.tasks:
            if node_id not in visited:
                visit(node_id)
                
        return order
        
    def get_waves(self) -> List[List[str]]:
        """Calculate execution waves for parallel scheduling.
        
        Groups tasks into "waves" that can be executed in parallel, where
        each wave depends only on previous waves.
        
        Returns:
            List of waves, where each wave is a list of node IDs
        """
        # Calculate node depths based on dependencies
        depths: Dict[str, int] = {}
        
        # Start with input nodes at depth 0
        for node_id in self.input_nodes:
            depths[node_id] = 0
            
        # Helper function to calculate depth
        def get_depth(node_id: str) -> int:
            """Calculate the depth of a node in the graph."""
            if node_id in depths:
                return depths[node_id]
                
            # Calculate as 1 + max depth of dependencies
            task = self.tasks[node_id]
            if not task.dependencies:
                depths[node_id] = 0
                return 0
                
            max_dep_depth = max(get_depth(dep_id) for dep_id in task.dependencies)
            depths[node_id] = max_dep_depth + 1
            return depths[node_id]
            
        # Calculate depths for all nodes
        for node_id in self.tasks:
            get_depth(node_id)
            
        # Group nodes by depth into waves
        waves: Dict[int, List[str]] = {}
        for node_id, depth in depths.items():
            if depth not in waves:
                waves[depth] = []
            waves[depth].append(node_id)
            
        # Convert to list of waves
        max_depth = max(waves.keys()) if waves else 0
        return [waves.get(depth, []) for depth in range(max_depth + 1)]


@dataclasses.dataclass
class ExecutionResult:
    """Result of executing a computation graph.
    
    Contains the outputs of each node in the graph as well as metrics
    about the execution.
    
    Attributes:
        node_outputs: Dictionary mapping node IDs to their outputs
        metrics: Execution metrics (timing, etc.)
        errors: Dictionary of errors encountered during execution
    """
    
    node_outputs: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    errors: Dict[str, Exception] = dataclasses.field(default_factory=dict)
    
    def get_result(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the result for a specific node.
        
        Args:
            node_id: ID of the node to retrieve results for
            
        Returns:
            Node's output or None if not found
        """
        return self.node_outputs.get(node_id)
        
    def get_error(self, node_id: str) -> Optional[Exception]:
        """Get the error for a specific node.
        
        Args:
            node_id: ID of the node to retrieve error for
            
        Returns:
            Node's error or None if no error occurred
        """
        return self.errors.get(node_id)
        
    def has_error(self) -> bool:
        """Check if any errors occurred during execution.
        
        Returns:
            True if at least one node had an error
        """
        return len(self.errors) > 0
        
    def is_complete(self) -> bool:
        """Check if execution completed without errors.
        
        Returns:
            True if execution completed successfully
        """
        return not self.has_error()
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update execution metrics.
        
        Args:
            metrics: New metrics to add
        """
        self.metrics.update(metrics)