"""
Structural JIT: Graph-Based Auto-Optimization for Ember Operators

Providing a just-in-time (JIT) compilation system for Ember operators
that analyzes operator structure directly rather than relying on execution tracing.
Converting operator compositions into optimized XCS graphs and executing
them with the appropriate scheduling strategy.

Key capabilities:
1. Structural analysis using Python's pytree protocol
2. Automatic graph construction without execution tracing
3. Parallel execution of independent operations
4. Adaptive scheduling based on graph structure

The implementation uses immutable data structures and side-effect-free functions
with a modular design:
- Components are focused on specific aspects of the JIT process
- New execution strategies can be added without modifying existing code
- Strategy implementations are interchangeable
- High-level modules depend on abstractions rather than specific implementations

Example:
    ```python
    @structural_jit
    class MyCompositeOperator(Operator):
        def __init__(self):
            self.op1 = SubOperator1()
            self.op2 = SubOperator2()

        def forward(self, *, inputs):
            # Multi-step computation
            intermediate = self.op1(inputs=inputs)
            result = self.op2(inputs=intermediate)
            return result

    # Using the optimized operator
    op = MyCompositeOperator()
    result = op(inputs={"text": "example"})
    # result == {"output": "processed example"}
    ```
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from ember.xcs.engine.xcs_engine import (
    IScheduler,
    TopologicalSchedulerWithParallelDispatch,
    compile_graph,
)
from ember.xcs.engine.xcs_noop_scheduler import XCSNoOpScheduler
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.tracer_decorator import JITCache

# Logger for this module
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")  # Generic return type
OpT = TypeVar("OpT", bound="Operator")  # Operator type

# Cache for compiled graphs
_structural_jit_cache = JITCache[XCSGraph]()


# -----------------------------------------------------------------------------
# Protocols & Type Definitions
# -----------------------------------------------------------------------------


@runtime_checkable
class Operator(Protocol):
    """Protocol defining the expected interface for Operators."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the operator with provided inputs."""
        pass


@runtime_checkable
class PytreeCompatible(Protocol):
    """Protocol for objects compatible with the pytree protocol."""

    def __pytree_flatten__(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Flatten object into a list of dynamic values and static metadata."""
        pass

    @classmethod
    def __pytree_unflatten__(cls, metadata: Dict[str, Any], values: List[Any]) -> Any:
        """Reconstruct object from flattened values and metadata."""
        pass


@runtime_checkable
class StructureDependency(Protocol):
    """Protocol for operators to declare structural dependencies.

    Operators implementing this protocol can explicitly define their
    structural dependencies, improving the precision of structural JIT
    and enabling state-aware caching.
    """

    def get_structural_dependencies(self) -> Dict[str, List[str]]:
        """Return mapping of operator attribute names to their dependencies.

        Returns:
            Dict mapping attribute names to lists of attribute names they depend on.
            Example: {"output_field": ["input_field1", "input_field2"]}
        """
        pass

    def get_structure_signature(self) -> str:
        """Return a signature representing the current structure state.

        When this signature changes, cached structure graphs should be invalidated.
        This could be a hash of structure variables or a version number that
        the operator increments when structure changes.

        Returns:
            A string signature representing the current structure state.
        """
        pass


# -----------------------------------------------------------------------------
# Execution Strategy Definition
# -----------------------------------------------------------------------------


@dataclass
class ExecutionConfig:
    """Configuration for graph execution.

    Defines parameters for scheduler selection and execution behavior.

    Attributes:
        strategy: Execution approach to use
        parallel_threshold: Minimum nodes to trigger parallelism in auto mode
        max_workers: Maximum concurrent worker threads for parallel execution
    """

    strategy: str = "auto"
    parallel_threshold: int = 5
    max_workers: Optional[int] = None


def get_scheduler(graph: XCSGraph, config: ExecutionConfig) -> IScheduler:
    """Create the appropriate scheduler based on strategy and graph.

    Analyzes graph characteristics and config settings to select
    the optimal scheduler implementation.

    Args:
        graph: Graph to be executed
        config: Execution configuration parameters

    Returns:
        Scheduler instance optimized for the graph

    Raises:
        ValueError: If strategy is invalid
    """
    # Handle pre-defined strategies first
    if config.strategy == "sequential":
        return XCSNoOpScheduler()

    if config.strategy == "parallel":
        return TopologicalSchedulerWithParallelDispatch(max_workers=config.max_workers)

    if config.strategy == "auto":
        # Auto mode - analyze graph for parallelization potential
        if len(graph.nodes) < config.parallel_threshold:
            return XCSNoOpScheduler()

        # Count potentially parallelizable nodes
        parallel_nodes = _count_parallelizable_nodes(graph)
        return (
            TopologicalSchedulerWithParallelDispatch(max_workers=config.max_workers)
            if parallel_nodes >= 2
            else XCSNoOpScheduler()
        )

    # Invalid strategy
    raise ValueError(
        f"Unknown execution strategy: {config.strategy}. "
        "Expected 'auto', 'parallel', or 'sequential'."
    )


def _count_parallelizable_nodes(graph: XCSGraph) -> int:
    """Count nodes that could execute in parallel.

    Analyzes graph structure to identify potential parallelism.

    Args:
        graph: Graph to analyze

    Returns:
        Estimated count of parallelizable nodes
    """
    # Count nodes with no dependencies (root nodes)
    root_nodes = sum(1 for node in graph.nodes.values() if not node.inbound_edges)
    if root_nodes > 1:
        return root_nodes

    # Count nodes with only one dependency (could execute in parallel after the root)
    return sum(1 for node in graph.nodes.values() if len(node.inbound_edges) == 1)


# -----------------------------------------------------------------------------
# Operator Structure Analysis
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorStructureNode:
    """
    Immutable representation of an operator in the structure graph.

    This class captures the essential information about an operator and its
    relationships to other operators in the composition structure.

    Attributes:
        operator: The actual operator instance
        node_id: Unique identifier for this node
        attribute_path: Dot-notation path to this operator from the root
        parent_id: ID of the parent node, or None for the root
        metadata: Dictionary for storing node-specific metadata
    """

    operator: Operator
    node_id: str
    attribute_path: str
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorStructureGraph:
    """
    Graph representation of an operator's composition structure.

    Captures the hierarchical structure of operator composition by analyzing
    the operator's attribute hierarchy through the pytree protocol.

    Attributes:
        nodes: Dictionary mapping node IDs to OperatorStructureNode instances
        root_id: ID of the root node in the graph
    """

    nodes: Dict[str, OperatorStructureNode] = field(default_factory=dict)
    root_id: Optional[str] = None


def _analyze_operator_structure(operator: Operator) -> OperatorStructureGraph:
    """Analyze operator composition structure.

    Identifies nested operators with their parent-child relationships.
    If the operator implements the StructureDependency protocol, uses
    the explicitly declared dependencies for more precise analysis.

    Args:
        operator: Root operator

    Returns:
        Operator structure graph
    """
    graph = OperatorStructureGraph()
    visited = set()
    logger = logging.getLogger("ember.xcs.tracer.structural_jit")

    # First check for explicit structural dependencies
    if isinstance(operator, StructureDependency) and hasattr(
        operator, "get_structural_dependencies"
    ):
        try:
            explicit_deps = operator.get_structural_dependencies()
            class_name = operator.__class__.__name__
            logger.debug(f"Using explicit structural dependencies for {class_name}")

            # Create the root node
            root_node_id = f"node_{id(operator)}"
            graph.nodes[root_node_id] = OperatorStructureNode(
                operator=operator, node_id=root_node_id, attribute_path="root"
            )
            graph.root_id = root_node_id

            # Add dependencies from the explicit declaration
            for attr_name, _ in explicit_deps.items():
                # Skip if we can't get the attribute
                if not hasattr(operator, attr_name):
                    continue

                # Add dependent operator if it exists
                attr_value = getattr(operator, attr_name)
                if isinstance(attr_value, Operator):
                    attr_node_id = f"node_{id(attr_value)}"
                    graph.nodes[attr_node_id] = OperatorStructureNode(
                        operator=attr_value,
                        node_id=attr_node_id,
                        attribute_path=f"root.{attr_name}",
                        parent_id=root_node_id,
                    )

            # If we successfully used explicit dependencies, return now
            if len(graph.nodes) > 1:  # More than just the root node
                return graph

            # Otherwise fall back to heuristic analysis
            logger.debug(
                "Explicit dependencies produced incomplete graph, "
                "falling back to heuristic analysis"
            )
        except Exception as e:
            logger.warning(
                f"Error using explicit structural dependencies: {e}. "
                f"Falling back to heuristic analysis."
            )

    # Reset graph for heuristic analysis
    graph = OperatorStructureGraph()
    visited = set()

    def visit(obj: Any, path: str, parent_id: Optional[str] = None) -> Optional[str]:
        """Recursively process object and its attributes.

        Args:
            obj: Current object
            path: Attribute path from root
            parent_id: Parent node ID

        Returns:
            Node ID if operator was added
        """
        # Skip primitives and None
        if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            return None

        # Skip cycles
        obj_id = id(obj)
        if obj_id in visited:
            return None

        # Mark as visited to prevent cycles
        visited.add(obj_id)

        # Add node if it's an operator
        node_id = None
        if isinstance(obj, Operator):
            node_id = f"node_{obj_id}"
            # Create the node
            node = OperatorStructureNode(
                operator=obj, node_id=node_id, attribute_path=path, parent_id=parent_id
            )

            # Capture type information from specification if available
            if hasattr(obj, "specification"):
                spec = obj.specification
                # Capture input model type for dict-to-model conversion
                if hasattr(spec, "input_model") and spec.input_model:
                    node.metadata["input_model"] = spec.input_model

                # Capture output model type for model-to-dict conversion
                if hasattr(spec, "structured_output") and spec.structured_output:
                    node.metadata["structured_output"] = spec.structured_output

            graph.nodes[node_id] = node

            # First node becomes root
            if graph.root_id is None:
                graph.root_id = node_id

        # Process attributes regardless of whether it's an operator
        # This is critical for nested operators!
        if hasattr(obj, "__dict__"):
            for attr_name, value in _get_attributes(obj):
                if attr_name.startswith("_"):
                    continue

                attr_path = f"{path}.{attr_name}"
                visit(value, attr_path, node_id or parent_id)

        # Process collections
        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(value, f"{path}[{key}]", node_id or parent_id)
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                visit(value, f"{path}[{i}]", node_id or parent_id)

        return node_id

    # Start traversal from root
    visit(operator, "root")
    return graph


def _get_attributes(obj: Any) -> List[Tuple[str, Any]]:
    """Get accessible attributes of an object.

    Extracts attributes that could potentially contain operators.

    Args:
        obj: Object to examine

    Returns:
        List of (name, value) tuples
    """
    # Start with instance variables
    attributes = list(getattr(obj, "__dict__", {}).items())

    # Add class variables for class objects
    if inspect.isclass(obj):
        for name in dir(obj):
            if not name.startswith("_"):
                try:
                    value = getattr(obj, name)
                    if not callable(value):
                        attributes.append((name, value))
                except Exception:
                    pass

    return attributes


# -----------------------------------------------------------------------------
# XCS Graph Building
# -----------------------------------------------------------------------------


def _build_xcs_graph_from_structure(
    operator: Operator,
    structure: OperatorStructureGraph,
    sample_input: Optional[Dict[str, Any]] = None,
) -> XCSGraph:
    """Build execution graph from operator structure.

    Creates a graph with nodes and edges based on the analyzed
    operator composition structure. Sets explicit output node metadata
    for deterministic result extraction.

    Args:
        operator: Root operator
        structure: Analyzed structure graph
        sample_input: Optional input for data flow analysis

    Returns:
        Executable XCS graph
    """
    graph = XCSGraph()

    # Add all operators as nodes with their metadata
    for node_id, node in structure.nodes.items():
        # Extract operator object
        operator_obj = node.operator

        # Add the node to the graph
        xcs_node = graph.add_node(operator=operator_obj, node_id=node_id)

        # Determine the actual metadata target
        metadata_target = None
        if not isinstance(xcs_node, str) and hasattr(xcs_node, "metadata"):
            metadata_target = xcs_node
        elif (
            isinstance(xcs_node, str)
            and node_id in graph.nodes
            and hasattr(graph.nodes[node_id], "metadata")
        ):
            metadata_target = graph.nodes[node_id]

        # Apply metadata if we have a valid target
        if metadata_target is not None:
            # Capture input_model from operator's specification
            if hasattr(operator_obj, "specification"):
                spec = operator_obj.specification
                if hasattr(spec, "input_model") and spec.input_model:
                    metadata_target.metadata["input_model"] = spec.input_model

            # Preserve any existing metadata
            if isinstance(node.metadata, dict):
                metadata_target.metadata.update(node.metadata)

    # Connect parent-child relationships
    for node_id, node in structure.nodes.items():
        if node.parent_id:
            graph.add_edge(from_id=node.parent_id, to_id=node_id)

    # Try to determine and set the output node for deterministic result extraction
    leaf_nodes = [
        node_id
        for node_id, node in structure.nodes.items()
        if node_id not in [edge.from_node for edge in graph.edges.values()]
    ]

    if leaf_nodes:
        # Use the root node if it's a leaf (single node graph)
        if structure.root_id in leaf_nodes:
            output_node_id = structure.root_id
        else:
            # Otherwise use the last leaf node in the list
            output_node_id = leaf_nodes[-1]

        # Set explicit output node ID metadata
        graph.metadata["output_node_id"] = output_node_id
        # Legacy metadata for backward compatibility
        graph.metadata["output_node"] = output_node_id

    return graph


# -----------------------------------------------------------------------------
# Execution & Caching
# -----------------------------------------------------------------------------


def _execute_with_engine(
    graph: XCSGraph,
    inputs: Dict[str, Any],
    config: ExecutionConfig,
) -> Dict[str, Any]:
    """Execute a graph using the XCS engine.

    Core execution method for structural JIT that handles graph execution
    with appropriate scheduling based on graph characteristics.

    Args:
        graph: Graph to execute
        inputs: Input data
        config: Execution configuration

    Returns:
        Execution results

    Raises:
        OperatorExecutionError: For errors in operator implementation
        Exception: For errors in graph execution machinery
    """
    logger = logging.getLogger("ember.xcs.tracer.structural_jit")

    # Get appropriate scheduler based on strategy and graph
    scheduler = get_scheduler(graph, config)
    scheduler_name = scheduler.__class__.__name__
    logger.debug(
        f"Executing graph with {len(graph.nodes)} nodes using {scheduler_name}"
    )

    try:
        # Compile and execute graph
        plan = compile_graph(graph=graph)
        results = scheduler.run_plan(
            plan=plan,
            global_input=inputs,
            graph=graph,
        )

        # Find appropriate output from results
        result = _extract_result(graph, results, logger)
        return result

    except Exception as e:
        # Handle execution errors
        from ember.core.exceptions import OperatorExecutionError

        # Propagate operator errors without recovery attempts
        if isinstance(e, (OperatorExecutionError, ValueError, TypeError, RuntimeError)):
            raise

        # For machinery errors, try to recover with cached result if available
        if hasattr(graph, "original_result") and graph.original_result is not None:
            logger.debug(f"Recovering from JIT error: {str(e)}")
            return graph.original_result

        # Cannot recover - re-raise the original exception
        raise


def _extract_result(
    graph: XCSGraph, results: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """Extract the appropriate result from graph execution output.

    Uses a deterministic prioritized approach to identify the output value.
    Applies explicit metadata markers first, falling back to structural analysis
    when explicit markers aren't available.

    Args:
        graph: The executed graph
        results: Execution results for all nodes
        logger: Logger for debug messages

    Returns:
        The extracted result
    """
    # Priority 1: Explicit metadata markers
    if "output_node_id" in graph.metadata:
        node_id = graph.metadata["output_node_id"]
        if node_id in results:
            logger.debug(f"Using explicit output_node_id: {node_id}")
            return results[node_id]
        logger.warning(f"Output node '{node_id}' not found in results")

    if "original_operator" in results:
        logger.debug("Using original_operator node")
        return results["original_operator"]

    if "output_node" in graph.metadata and graph.metadata["output_node"] in results:
        node_id = graph.metadata["output_node"]
        logger.debug(f"Using legacy output_node: {node_id}")
        return results[node_id]

    # Priority 2: Structural inference
    # Simple case: single node graph
    if len(graph.nodes) == 1:
        node_id = next(iter(graph.nodes.keys()))
        if node_id in results:
            logger.debug(f"Using only node: {node_id}")
            return results[node_id]

    # Get leaf nodes (terminal outputs)
    leaf_nodes = [
        node_id for node_id, node in graph.nodes.items() if not node.outbound_edges
    ]

    # Single leaf node is unambiguous
    if len(leaf_nodes) == 1 and leaf_nodes[0] in results:
        logger.debug(f"Using single leaf node: {leaf_nodes[0]}")
        return results[leaf_nodes[0]]

    # Multiple identical leaf results
    if leaf_nodes:
        available_results = [
            (node, results[node]) for node in leaf_nodes if node in results
        ]
        if available_results and all(
            r[1] == available_results[0][1] for r in available_results
        ):
            logger.debug(
                f"Using identical result from {len(available_results)} leaf nodes"
            )
            return available_results[0][1]

    # Priority 3: Recovery mechanisms
    if hasattr(graph, "original_result") and graph.original_result is not None:
        logger.debug("Using cached original result")
        return graph.original_result

    # Last resort: return all results
    logger.debug("Could not determine specific output node, returning all results")
    return results


# -----------------------------------------------------------------------------
# JIT Decorator Implementation
# -----------------------------------------------------------------------------


def structural_jit(
    func: Optional[Type[OpT]] = None,
    *,
    execution_strategy: str = "auto",
    parallel_threshold: int = 5,
    max_workers: Optional[int] = None,
    cache_graph: bool = True,
) -> Union[Callable[[Type[OpT]], Type[OpT]], Type[OpT]]:
    """Structure-based JIT optimization for operators.

    Analyzes operator composition structure to build optimized execution graphs
    without runtime tracing. Automatically identifies parallelization opportunities.

    Args:
        func: Operator class to decorate
        execution_strategy: Execution approach:
            - "auto": Select based on graph analysis
            - "parallel": Force parallel execution
            - "sequential": Force sequential execution
        parallel_threshold: Minimum nodes for parallelization in auto mode
        max_workers: Maximum concurrent workers for parallel execution
        cache_graph: Whether to cache graphs for repeated execution

    Returns:
        Decorated operator class with optimized execution

    Example:
        ```python
        @structural_jit
        class MyOperator(Operator):
            def __init__(self):
                self.op1 = SubOperator1()
                self.op2 = SubOperator2()

            def forward(self, *, inputs):
                intermediate = self.op1(inputs=inputs)
                return self.op2(inputs=intermediate)
        ```
    """

    def decorator(cls: Type[OpT]) -> Type[OpT]:
        """Inner decorator applied to operator class."""
        # Verify interface compatibility
        if not callable(cls) or not callable(cls.__call__):
            raise TypeError("@structural_jit requires a class with __call__ method")

        # Create execution config once
        execution_config = ExecutionConfig(
            strategy=execution_strategy,
            parallel_threshold=parallel_threshold,
            max_workers=max_workers,
        )

        # Save original methods
        original_init = cls.__init__
        original_call = cls.__call__

        @functools.wraps(original_init)
        def init_wrapper(self: OpT, *args: Any, **kwargs: Any) -> None:
            """Wrapped initialization with structure analysis."""
            # Initialize operator
            original_init(self, *args, **kwargs)

            # JIT properties
            self._jit_enabled = True
            self._jit_config = execution_config
            self._jit_cache_graph = cache_graph

            # Analyze structure during initialization
            self._jit_structure_graph = _analyze_operator_structure(self)
            self._jit_xcs_graph = None

        @functools.wraps(original_call)
        def call_wrapper(self: OpT, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Wrapped execution with graph-based optimization."""
            # Handle disabled JIT
            if getattr(self, "_jit_enabled", True) is False:
                return original_call(self, inputs=inputs)

            # Prevent infinite recursion
            if getattr(self, "_jit_in_execution", False):
                return original_call(self, inputs=inputs)

            try:
                # Set recursion guard
                self._jit_in_execution = True

                # Get state signature if available
                state_signature = None
                if isinstance(self, StructureDependency) and hasattr(
                    self, "get_structure_signature"
                ):
                    try:
                        state_signature = self.get_structure_signature()
                    except Exception as e:
                        logger.warning(f"Error getting structure signature: {e}")

                # Try to get cached graph with state validation
                graph = None
                if self._jit_cache_graph:
                    graph = _structural_jit_cache.get_with_state(self, state_signature)

                # Use cached graph if available
                if graph is not None:
                    # Measure execution time for metrics
                    execution_start = time.time()
                    result = _execute_with_engine(
                        graph=graph,
                        inputs=inputs,
                        config=self._jit_config,
                    )
                    execution_duration = time.time() - execution_start
                    _structural_jit_cache.metrics.record_execution(execution_duration)
                    return result

                # First call - build the graph
                # Get original results
                original_result = original_call(self, inputs=inputs)

                # Measure compilation time for metrics
                compilation_start = time.time()

                # Build and configure graph
                structure = self._jit_structure_graph
                if structure is None:
                    # Just in case structure wasn't analyzed during init
                    structure = _analyze_operator_structure(self)
                    self._jit_structure_graph = structure

                graph = _build_xcs_graph_from_structure(
                    operator=self,
                    structure=structure,
                    sample_input=inputs,
                )

                # Save original result and add original operator node
                graph.original_result = original_result
                graph.add_node(
                    operator=original_call.__get__(self),
                    node_id="original_operator",
                )

                # Record compilation time
                compilation_duration = time.time() - compilation_start
                _structural_jit_cache.metrics.record_compilation(compilation_duration)

                # Cache the graph with state signature
                if self._jit_cache_graph:
                    _structural_jit_cache.set(self, graph, state_signature)

                # Update instance variable for backward compatibility
                self._jit_xcs_graph = graph

                return original_result
            finally:
                self._jit_in_execution = False

        # Replace methods with wrapped versions
        cls.__init__ = cast(Callable, init_wrapper)
        cls.__call__ = cast(Callable, call_wrapper)

        # Add control utilities
        cls.disable_jit = lambda self: setattr(self, "_jit_enabled", False)
        cls.enable_jit = lambda self: setattr(self, "_jit_enabled", True)
        cls.clear_graph_cache = lambda self: (
            _structural_jit_cache.invalidate(self),
            setattr(self, "_jit_xcs_graph", None),
        )
        cls.get_jit_metrics = lambda self: _structural_jit_cache.get_metrics()

        return cls

    # Handle both @structural_jit and @structural_jit(...) syntax
    return decorator(func) if func is not None else decorator


# -----------------------------------------------------------------------------
# Context Manager for Testing
# -----------------------------------------------------------------------------


@contextmanager
def disable_structural_jit() -> None:
    """
    Context manager that temporarily disables structural JIT for testing.

    This utility is primarily intended for testing and debugging scenarios
    where you need to compare behavior with and without JIT optimization.

    Example:
        with disable_structural_jit():
            # JIT-decorated operators will run without optimization
            result = my_operator(inputs=test_input)
    """
    # Save all decorated operators we find
    operators = []

    # Find all objects in memory that have _jit_enabled attribute
    import gc

    for obj in gc.get_objects():
        if hasattr(obj, "_jit_enabled"):
            operators.append(obj)
            obj._jit_enabled = False

    try:
        yield
    finally:
        # Restore JIT state
        for op in operators:
            op._jit_enabled = True
