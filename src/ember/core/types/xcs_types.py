"""
Type definitions for XCS (eXecution Control System) components.

This module provides type-safe definitions for XCS components,
including Graph, Plan, Task, and other execution-related types.

The type system follows these design principles:
1. All interfaces are explicitly defined with clear contracts
2. Type variance is handled appropriately for input/output types
3. Type parameters are clearly documented and consistently named
4. Runtime checkable protocols are provided for easier debugging
5. All collection types are precisely defined with appropriate bounds
"""

from typing import (
    Dict, 
    List,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Type,
    Generic,
    cast,
    Hashable,
    Any,
    ClassVar,
    final,
    overload,
)
from typing_extensions import TypedDict, NotRequired, Literal, Protocol as ExtProtocol

# Standard invariant type variables for regular usage
NodeInputT = TypeVar("NodeInputT", bound=Mapping[str, object])
NodeOutputT = TypeVar("NodeOutputT", bound=Mapping[str, object])
OperatorT = TypeVar("OperatorT", bound=Callable[..., object])


@final
class NodeMetadata(TypedDict, total=False):
    """
    TypedDict for node metadata with standardized fields.
    
    These fields are optional and provide additional context about a node
    in the execution graph. Extension is possible through the custom_data field.
    """
    source_file: str  # Path to the source file where the node was defined
    source_line: int  # Line number in source where the node was defined
    author: str       # Author or owner of the node
    version: str      # Version information (semantic versioning recommended)
    created_at: str   # ISO-8601 timestamp of creation
    updated_at: str   # ISO-8601 timestamp of last update
    description: str  # Human-readable description of the node's purpose
    custom_data: Dict[str, object]  # Extension point for additional metadata


@final
class XCSNodeAttributes(TypedDict, total=False):
    """
    Attributes that can be attached to a node in an XCS graph.
    
    This provides a strongly-typed structure for node attributes
    while allowing for extension with additional fields.
    """
    name: str                # Human-readable name for the node
    description: str         # Detailed description of node functionality
    tags: List[str]          # Tags for organization/filtering
    metadata: NodeMetadata   # Structured metadata about the node


@final
class ResultMetadata(TypedDict, total=False):
    """
    TypedDict for execution result metadata with standardized fields.
    
    These fields capture performance and execution context information
    that is useful for monitoring, debugging, and optimization.
    """
    start_time: float        # Unix timestamp when execution started
    end_time: float          # Unix timestamp when execution completed
    execution_time: float    # Duration in seconds (may differ from end-start due to precision)
    memory_usage: int        # Peak memory usage in bytes
    cpu_usage: float         # CPU usage percentage (0.0-100.0)
    device: str              # Device ID/name where execution occurred 
    custom_data: Dict[str, object]  # Extension point for additional metrics


@final
class XCSNodeResult(TypedDict, Generic[NodeOutputT]):
    """
    Result of executing a node in an XCS graph.
    
    This provides a strongly-typed structure for execution results
    with clear semantics for success/failure handling. The result field
    is strongly typed to match the node's output type.
    
    Generic parameters:
        NodeOutputT: Type of outputs this node produces
    """
    success: bool                         # Whether execution succeeded
    result: Optional[NodeOutputT]         # Output on success, None on failure
    error: NotRequired[str]               # Error message on failure 
    execution_time: NotRequired[float]    # Execution duration in seconds
    metadata: NotRequired[ResultMetadata] # Additional execution metadata


# Define the protocol version for runtime type checking
@runtime_checkable
class XCSNode(Protocol[NodeInputT, NodeOutputT]):  # type: ignore
    """
    Protocol interface for an XCS graph node.
    
    A node is the fundamental unit of execution in the XCS system.
    It accepts typed inputs, performs computation via its operator,
    and produces typed outputs. Nodes can be connected to form a
    directed graph representing computation flow.
    
    Generic parameters:
        NodeInputT: Type of inputs this node accepts
        NodeOutputT: Type of outputs this node produces
        
    Note: type: ignore is used to suppress a mypy error related to variance in protocols.
          In an ideal world with Python 3.12+, we'd use explicit variance annotations to fix this.
    """
    node_id: str                          # Unique identifier for this node 
    operator: Callable[[NodeInputT], NodeOutputT]  # Function to execute
    inbound_edges: List[str]              # IDs of nodes that feed into this node
    outbound_edges: List[str]             # IDs of nodes this node feeds into
    attributes: XCSNodeAttributes         # Additional node metadata
    captured_outputs: Optional[NodeOutputT]  # Cached result from last execution

# Concrete class for XCSNode for better type compatibility with mypy
class XCSNodeImpl(Generic[NodeInputT, NodeOutputT]):
    """
    Concrete implementation of XCSNode interface.
    
    This implementation provides a base class that can be extended
    for concrete node implementations with strong typing support.
    
    Generic parameters:
        NodeInputT: Type of inputs this node accepts
        NodeOutputT: Type of outputs this node produces
    """
    def __init__(
        self, 
        node_id: str, 
        operator: Callable[[NodeInputT], NodeOutputT],
        attributes: Optional[XCSNodeAttributes] = None
    ):
        self.node_id = node_id
        self.operator = operator
        self.inbound_edges: List[str] = []
        self.outbound_edges: List[str] = []
        self.attributes: XCSNodeAttributes = attributes or {}
        self.captured_outputs: Optional[NodeOutputT] = None


@final
class XCSTaskStatus(TypedDict):
    """
    Status information for an XCS execution task.
    
    This structure tracks the runtime state of a task during execution,
    including progress, timing, and error information.
    """
    state: Literal["pending", "running", "completed", "failed", "cancelled"]  # Current state
    progress: float                # Completion percentage (0.0 to 1.0)
    start_time: Optional[float]    # Unix timestamp when task started, None if not started
    end_time: Optional[float]      # Unix timestamp when task completed/failed, None if not finished
    retries: int                   # Current retry count (0 if first attempt)
    error_message: Optional[str]   # Error details if failed, None otherwise


@final
class XCSTaskDefinition(TypedDict):
    """
    Definition of an executable task in the XCS execution plan.
    
    A task represents a schedulable unit of work in the execution system.
    Each task corresponds to a node in the original computation graph,
    with additional execution parameters and dependency information.
    """
    node_id: str                   # ID of the node this task will execute
    dependencies: List[str]        # Task IDs that must complete before this task
    priority: int                  # Scheduling priority (higher values = higher priority)
    timeout_seconds: Optional[float]  # Maximum execution time before timeout, None for no limit
    retry_policy: Literal["none", "linear", "exponential"]  # Backoff strategy for retries
    max_retries: int               # Maximum retry attempts (0 = no retries)


# Define the protocol version for runtime type checking
@runtime_checkable
class XCSGraph(Protocol[NodeInputT, NodeOutputT]):
    """
    Protocol interface for an XCS computation graph.
    
    A computation graph represents a directed graph where nodes are
    operators and edges represent data flow. The graph provides methods
    for constructing and inspecting the computational structure.
    
    Generic parameters:
        NodeInputT: Type of inputs nodes in this graph accept
        NodeOutputT: Type of outputs nodes in this graph produce
    """
    nodes: Dict[str, XCSNode[NodeInputT, NodeOutputT]]  # All nodes in this graph, indexed by ID

    def add_node(
        self, 
        node_id: str, 
        operator: Callable[[NodeInputT], NodeOutputT], 
        **attributes: object
    ) -> None:
        """
        Add a node to the graph.
        
        Creates a new node with the given ID and operator function, 
        with optional attributes for metadata and configuration.
        
        Args:
            node_id: Unique identifier for the node
            operator: Callable that will be executed when this node runs
            **attributes: Additional attributes to attach to the node
                
        Raises:
            ValueError: If node_id already exists in the graph
        """
        ...

    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add a directed edge between nodes.
        
        Creates a data flow connection from the source node to the
        destination node. This indicates that the output of from_node
        will be used as input to to_node during execution.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
            
        Raises:
            ValueError: If either node doesn't exist in the graph
        """
        ...

    def get_node(self, node_id: str) -> XCSNode[NodeInputT, NodeOutputT]:
        """
        Get a node by ID.
        
        Retrieves a node from the graph by its unique identifier.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            The requested node
        
        Raises:
            KeyError: If node with given ID doesn't exist
        """
        ...

# Concrete class for better type compatibility with mypy
class XCSGraphImpl(Generic[NodeInputT, NodeOutputT]):
    """
    Concrete implementation of XCSGraph interface.
    
    This implementation provides a base class that can be extended
    for concrete graph implementations with strong typing support.
    
    Generic parameters:
        NodeInputT: Type of inputs nodes in this graph accept
        NodeOutputT: Type of outputs nodes in this graph produce
    """
    def __init__(self) -> None:
        self.nodes: Dict[str, XCSNodeImpl[NodeInputT, NodeOutputT]] = {}
        
    def add_node(
        self, 
        node_id: str, 
        operator: Callable[[NodeInputT], NodeOutputT], 
        **attributes: object
    ) -> None:
        """Add a node to the graph."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in the graph")
            
        node = XCSNodeImpl(node_id=node_id, operator=operator, attributes=cast(XCSNodeAttributes, attributes))
        self.nodes[node_id] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge between nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Nodes {from_node} or {to_node} not found")

        self.nodes[from_node].outbound_edges.append(to_node)
        self.nodes[to_node].inbound_edges.append(from_node)

    def get_node(self, node_id: str) -> XCSNodeImpl[NodeInputT, NodeOutputT]:
        """Get a node by ID."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        return self.nodes[node_id]


# Define the protocol version for runtime type checking
@runtime_checkable
class XCSPlan(Protocol[NodeInputT, NodeOutputT]):
    """
    Protocol interface for an XCS execution plan.
    
    An execution plan transforms a computation graph into a sequence of
    executable tasks with dependencies, optimizing for parallel execution
    and resource utilization. It serves as the bridge between the abstract
    computation graph and the concrete execution engine.
    
    Generic parameters:
        NodeInputT: Type of inputs nodes in this graph accept
        NodeOutputT: Type of outputs nodes in this graph produce
    """
    tasks: Dict[str, XCSTaskDefinition]  # All tasks in this plan, indexed by ID
    original_graph: XCSGraph[NodeInputT, NodeOutputT]  # Reference to the source graph
    
    def get_execution_order(self) -> List[str]:
        """
        Get the topologically sorted execution order of tasks.
        
        Computes a valid sequential execution order for all tasks that respects
        the dependency relationships. When multiple execution orders are valid,
        the implementation may use heuristics to optimize for factors like
        execution time, memory usage, or resource utilization.
        
        Returns:
            List of task IDs in a valid execution order
            
        Raises:
            ValueError: If the dependency graph contains cycles
        """
        ...
        
# Concrete class for better type compatibility with mypy
class XCSPlanImpl(Generic[NodeInputT, NodeOutputT]):
    """
    Concrete implementation of XCSPlan interface.
    
    This implementation provides a base class that can be extended
    for concrete execution plan implementations with strong typing support.
    
    Generic parameters:
        NodeInputT: Type of inputs nodes in this graph accept
        NodeOutputT: Type of outputs nodes in this graph produce
    """
    def __init__(
        self,
        tasks: Dict[str, XCSTaskDefinition],
        original_graph: XCSGraph[NodeInputT, NodeOutputT]
    ):
        self.tasks = tasks
        self.original_graph = original_graph
        
    def get_execution_order(self) -> List[str]:
        """Get the topologically sorted execution order of tasks."""
        # Simple implementation - return tasks in arbitrary order
        # A real implementation would do topological sorting based on dependencies
        return list(self.tasks.keys())
