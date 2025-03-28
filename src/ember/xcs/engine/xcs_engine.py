"""XCS execution engine for computational graphs.

Core execution infrastructure for computational graphs in Ember. Provides 
mechanisms for compiling, scheduling, and dispatching graph-based computations
across different execution contexts.

This engine forms the foundation for higher-level optimizations including:
1. Just-in-Time (JIT) compilation via tracer_decorator.jit
2. Structure-based parallelization via structural_jit
3. Graph-based execution via autograph

Usage patterns:
1. Direct execution of computational graphs:
   ```python
   graph = XCSGraph()
   node1 = graph.add_node(preprocess_fn)
   node2 = graph.add_node(compute_fn)
   graph.add_edge(node1, node2)
   
   results = execute(graph, inputs={"data": input_data})
   ```

2. With execution options for performance tuning:
   ```python
   from ember.xcs.engine.execution_options import execution_options
   
   with execution_options(scheduler="parallel", max_workers=4):
       # All graph executions in this context use parallel execution
       results = my_jit_operator(inputs=data)
   ```

3. As the backend for JIT compilation:
   ```python
   @jit  # JIT uses XCS engine for compiled executions
   class MyOperator(Operator):
       def forward(self, *, inputs):
           return process(inputs)
   ```
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
)

from ember.xcs.graph.xcs_graph import XCSGraph

logger = logging.getLogger(__name__)


@dataclass
class XCSTask:
    """Executable task within a computation plan.

    Represents a single operation to be executed as part of a larger
    computational graph, along with its input and output relationships.

    Attributes:
        operator: Function or operator that performs the computation
        inputs: Node IDs that provide inputs to this task
        outputs: Node IDs that consume output from this task
    """

    operator: Callable[..., Dict[str, Any]]
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class XCSPlan:
    """Compiled execution plan for a computation graph.

    Transforms a graph description into a concrete execution plan with
    tasks that can be dispatched by a scheduler. Acts as an intermediate
    representation between graph definition and execution.

    Attributes:
        tasks: Mapping from node IDs to executable task objects
        graph_id: Unique identifier for tracking plan instances
    """

    tasks: Dict[str, XCSTask] = field(default_factory=dict)
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_task(self, node_id: str, task: XCSTask) -> None:
        """Registers a task in the execution plan.

        Args:
            node_id: Unique identifier for the task
            task: The executable task to register
        """
        self.tasks[node_id] = task


class IScheduler(Protocol):
    """Interface for graph execution schedulers.

    Defines the contract for schedulers that organize graph nodes into
    execution waves based on their dependencies. Different implementations
    can prioritize different execution strategies (sequential, parallel, etc).
    """

    def schedule(self, graph: XCSGraph) -> List[List[str]]:
        """Creates an execution plan from a computation graph.

        Args:
            graph: Computation graph to be scheduled

        Returns:
            List of execution waves, where each wave contains node IDs
            that can be executed concurrently
        """
        ...


class TopologicalScheduler:
    """Sequential scheduler based on dependency ordering.

    Organizes nodes into execution waves where each wave contains nodes
    whose dependencies have been satisfied by previous waves. Creates
    a valid execution order that respects data dependencies.
    """

    def schedule(self, graph: XCSGraph) -> List[List[str]]:
        """Groups nodes into dependency-respecting execution waves.

        Analyzes the graph structure to determine which nodes can be
        executed concurrently without violating data dependencies,
        organizing them into sequential waves.

        Args:
            graph: Computation graph to be scheduled

        Returns:
            List of waves where each wave contains nodes that can be
            executed in parallel after all previous waves complete

        Raises:
            ValueError: If graph contains cycles
        """
        # Get topological order
        topo_order = graph.topological_sort()

        # Organize into waves based on dependencies
        waves: List[List[str]] = []
        completed_nodes: Set[str] = set()

        while topo_order:
            # Find all nodes whose dependencies are satisfied
            current_wave = []
            remaining = []

            for node_id in topo_order:
                node = graph.nodes[node_id]
                if all(dep in completed_nodes for dep in node.inbound_edges):
                    current_wave.append(node_id)
                else:
                    remaining.append(node_id)

            # Add the wave and update completed nodes
            waves.append(current_wave)
            completed_nodes.update(current_wave)
            topo_order = remaining

        return waves


class TopologicalSchedulerWithParallelDispatch(TopologicalScheduler):
    """Parallel execution scheduler using multi-threading.

    Extends the topological scheduler with parallel dispatch capabilities,
    executing each wave of nodes concurrently using a thread pool. Automatically
    adapts execution based on identified parallelization patterns in the graph.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize with optional worker count limit.

        Args:
            max_workers: Maximum number of worker threads to use for each wave.
                         If None, uses number of nodes in the wave.
        """
        self.max_workers = max_workers
        super().__init__()

    def run_plan(
        self, *, plan: XCSPlan, global_input: Dict[str, Any], graph: XCSGraph
    ) -> Dict[str, Any]:
        """Execute a compiled plan using parallel dispatching.

        Executes a compiled plan with optimized parallelism strategies,
        automatically detecting parallelizable patterns in the graph metadata
        and adjusting thread allocation accordingly.

        Args:
            plan: The XCSPlan to execute
            global_input: Input data for the graph
            graph: The source graph (used for dependency analysis and metadata)

        Returns:
            A dictionary mapping node IDs to their execution results

        Raises:
            Exception: Propagates exceptions from individual node executions
                      after logging them
        """
        results: Dict[str, Dict[str, Any]] = {}
        logger.debug(f"Running plan with {len(plan.tasks)} tasks")

        # Get the waves by running schedule on the graph
        waves = self.schedule(graph)
        logger.debug(f"Execution plan with {len(waves)} waves")

        # Check for known parallel patterns in the graph metadata
        parallel_info = {}
        if hasattr(graph, "metadata") and graph.metadata:
            # Extract parallelization information
            parallelizable_nodes = graph.metadata.get("parallelizable_nodes", [])
            aggregator_nodes = graph.metadata.get("aggregator_nodes", [])
            parallel_groups = graph.metadata.get("parallel_groups", {})

            if parallelizable_nodes:
                logger.debug(f"Found {len(parallelizable_nodes)} parallelizable nodes")
                parallel_info["parallelizable"] = set(parallelizable_nodes)

            if aggregator_nodes:
                logger.debug(f"Found {len(aggregator_nodes)} aggregator nodes")
                parallel_info["aggregators"] = set(aggregator_nodes)

            if parallel_groups:
                logger.debug(f"Found {len(parallel_groups)} parallel groups")
                parallel_info["groups"] = parallel_groups

        # Execute each wave in parallel
        for wave_idx, wave in enumerate(waves):
            logger.debug(
                f"Executing wave {wave_idx+1}/{len(waves)} with {len(wave)} nodes"
            )

            # Check if this wave contains parallelizable operations
            wave_is_parallelizable = parallel_info and any(
                node_id in parallel_info.get("parallelizable", set())
                for node_id in wave
            )

            # Optimize max_workers for this wave if we have parallelization info
            wave_max_workers = min(len(wave), self.max_workers or len(wave))
            if wave_is_parallelizable and wave_max_workers > 1:
                logger.debug(
                    f"Wave {wave_idx+1} contains parallelizable operations - using {wave_max_workers} workers"
                )

            # Create a thread pool for parallel execution with appropriate worker count
            with ThreadPoolExecutor(max_workers=wave_max_workers) as executor:
                futures = {}

                # Submit each node in the wave for execution
                for node_id in wave:
                    if node_id in plan.tasks:
                        task = plan.tasks[node_id]

                        # Collect inputs from predecessors
                        # Check if input is a dictionary or other type
                        try:
                            # Handle both Pydantic models and regular dictionaries
                            if hasattr(global_input, "model_copy"):
                                inputs = global_input.model_copy()
                            else:
                                inputs = global_input.copy()

                            for pred_id in task.inputs:
                                if pred_id in results:
                                    if hasattr(inputs, "update") and callable(
                                        inputs.update
                                    ):
                                        inputs.update(results[pred_id])
                                    else:
                                        # For non-dictionary inputs, we can't update them
                                        # Just pass through global_input
                                        inputs = global_input
                                        break
                        except (AttributeError, TypeError):
                            # If copy or update fails, just use the original input
                            inputs = global_input

                        # Add node name to inputs for tracking
                        if "node_name" in global_input:
                            inputs["node_name"] = node_id

                        # Add execution context for patterns like ensemble-judge
                        # This helps operators adapt their behavior for parallel execution
                        if parallel_info:
                            # Check if this node is part of a known parallel pattern
                            is_parallelizable = node_id in parallel_info.get(
                                "parallelizable", set()
                            )
                            is_aggregator = node_id in parallel_info.get(
                                "aggregators", set()
                            )

                            if is_parallelizable or is_aggregator:
                                # Add execution hints to the input
                                if "execution_context" not in inputs:
                                    inputs["execution_context"] = {}

                                inputs["execution_context"][
                                    "parallelizable"
                                ] = is_parallelizable
                                inputs["execution_context"][
                                    "aggregator"
                                ] = is_aggregator

                                # For aggregators, also add info about which group it aggregates
                                if is_aggregator and "data_flow" in graph.metadata:
                                    data_flow = graph.metadata["data_flow"]
                                    if (
                                        node_id in data_flow
                                        and "aggregates_groups" in data_flow[node_id]
                                    ):
                                        inputs["execution_context"][
                                            "aggregates"
                                        ] = list(
                                            data_flow[node_id][
                                                "aggregates_groups"
                                            ].keys()
                                        )

                        # Submit for execution
                        futures[executor.submit(task.operator, inputs=inputs)] = node_id

                # Collect results
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        results[node_id] = future.result()
                    except Exception as e:
                        logger.exception(f"Error executing node {node_id}: {e}")
                        results[node_id] = {"error": str(e)}

        return results


def execute_graph(
    graph: XCSGraph,
    global_input: Dict[str, Any],
    scheduler: Optional[IScheduler] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute a computational graph with automatic scheduling.

    Primary entry point for graph execution in XCS. Processes the graph by
    automatically detecting execution patterns, scheduling nodes into waves
    based on dependencies, and dispatching the computation across those waves.

    Args:
        graph: The computational graph to execute
        global_input: Input data for the graph's source nodes
        scheduler: Optional scheduler to determine execution order and strategy;
                  defaults to sequential TopologicalScheduler if not specified

    Returns:
        A dictionary mapping node IDs to their execution results

    Raises:
        ValueError: If the graph contains cycles or other invalid structures
    """
    if scheduler is None:
        scheduler = TopologicalScheduler()

    # Schedule the nodes
    waves = scheduler.schedule(graph)

    # Execute the graph
    results: Dict[str, Dict[str, Any]] = {}

    for wave in waves:
        # Single-threaded execution for simple scheduler
        if isinstance(scheduler, TopologicalScheduler) and not isinstance(
            scheduler, TopologicalSchedulerWithParallelDispatch
        ):
            for node_id in wave:
                node = graph.nodes[node_id]

                # Collect inputs from predecessors or use global input for source nodes
                try:
                    if not node.inbound_edges:
                        # Source node: use global input
                        inputs = (
                            global_input.copy()
                            if hasattr(global_input, "copy")
                            else global_input
                        )
                    else:
                        # Check if graph supports field-level mappings
                        if hasattr(graph, "prepare_node_inputs") and callable(
                            graph.prepare_node_inputs
                        ):
                            # Use field-level mapping for precise data flow
                            inputs = graph.prepare_node_inputs(node_id, results)
                        else:
                            # Fallback to legacy behavior: merge all predecessor outputs
                            inputs = {}
                            for pred_id in node.inbound_edges:
                                if hasattr(inputs, "update") and callable(
                                    inputs.update
                                ):
                                    inputs.update(results[pred_id])
                                else:
                                    # For non-dictionary inputs, set to the first result
                                    inputs = results[pred_id]
                                    break
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Error preparing inputs for node {node_id}: {e}")
                    # If prepare_node_inputs or update fails, use global_input for source nodes
                    # or first predecessor result for non-source nodes
                    inputs = (
                        global_input
                        if not node.inbound_edges
                        else results[node.inbound_edges[0]]
                    )

                # Add node_id to inputs for tracking in test functions
                if "node_name" in global_input:
                    inputs["node_name"] = node_id

                # Execute the node
                try:
                    # Auto-convert dict to EmberModel if needed - clean, minimal, type-safe
                    if isinstance(inputs, dict) and "input_model" in node.metadata:
                        input_model = node.metadata["input_model"]
                        if hasattr(input_model, "from_dict"):
                            inputs = input_model.from_dict(inputs)

                    node_result = node.operator(inputs=inputs)
                    results[node_id] = node_result
                except Exception as e:
                    logger.exception(f"Error executing node {node_id}: {e}")
                    results[node_id] = {"error": str(e)}

        # Parallel execution for parallel scheduler
        else:
            with ThreadPoolExecutor(
                max_workers=getattr(scheduler, "max_workers", None)
            ) as executor:
                futures = {}

                # Start all jobs in the wave
                for node_id in wave:
                    node = graph.nodes[node_id]

                    # Collect inputs from predecessors or use global input for source nodes
                    try:
                        if not node.inbound_edges:
                            # Source node: use global input
                            inputs = (
                                global_input.copy()
                                if hasattr(global_input, "copy")
                                else global_input
                            )
                        else:
                            # Check if graph supports field-level mappings
                            if hasattr(graph, "prepare_node_inputs") and callable(
                                graph.prepare_node_inputs
                            ):
                                # Use field-level mapping for precise data flow
                                inputs = graph.prepare_node_inputs(node_id, results)
                            else:
                                # Fallback to legacy behavior: merge all predecessor outputs
                                inputs = {}
                                for pred_id in node.inbound_edges:
                                    if hasattr(inputs, "update") and callable(
                                        inputs.update
                                    ):
                                        inputs.update(results[pred_id])
                                    else:
                                        # For non-dictionary inputs, set to the first result
                                        inputs = results[pred_id]
                                        break
                    except (AttributeError, TypeError) as e:
                        logger.warning(
                            f"Error preparing inputs for node {node_id}: {e}"
                        )
                        # If prepare_node_inputs or update fails, use global_input for source nodes
                        # or first predecessor result for non-source nodes
                        inputs = (
                            global_input
                            if not node.inbound_edges
                            else results[node.inbound_edges[0]]
                        )

                    # Add node_id to inputs for tracking in test functions
                    if "node_name" in global_input:
                        inputs["node_name"] = node_id

                    # Auto-convert dict to EmberModel if needed - clean, minimal, type-safe
                    if isinstance(inputs, dict) and "input_model" in node.metadata:
                        input_model = node.metadata["input_model"]
                        if hasattr(input_model, "from_dict"):
                            inputs = input_model.from_dict(inputs)

                    # Submit the job
                    futures[executor.submit(node.operator, inputs=inputs)] = node_id

                # Collect results
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        results[node_id] = future.result()
                    except Exception as e:
                        logger.exception(f"Error executing node {node_id}: {e}")
                        results[node_id] = {"error": str(e)}

    return results


def compile_graph(graph: XCSGraph) -> XCSPlan:
    """Compile a graph into an optimized execution plan.

    Transforms a graph definition into a concrete execution plan by analyzing
    dependencies and creating executable tasks. This intermediate representation
    separates graph definition from execution concerns, enabling optimization
    and specialization for different runtime environments.

    Args:
        graph: The XCS graph to compile

    Returns:
        An XCSPlan ready for execution by a scheduler

    Raises:
        ValueError: If the graph contains cycles or invalid nodes that would
                   prevent proper execution
    """
    # Ensure graph is valid with a topological sort
    topo_order = graph.topological_sort()

    # Create a new plan
    plan = XCSPlan()

    # Convert each node into a task with proper inputs and outputs
    for node_id in topo_order:
        node = graph.nodes[node_id]

        # Create a task for this node
        task = XCSTask(
            operator=node.operator,
            inputs=node.inbound_edges.copy(),
            outputs=node.outbound_edges.copy(),
        )

        # Add the task to the plan
        plan.add_task(node_id, task)

    return plan
