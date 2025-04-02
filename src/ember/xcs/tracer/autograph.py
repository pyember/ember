"""Dependency analysis and graph construction for XCS computational graphs.

Identifies data flow, state dependencies, and execution constraints between operator
invocations. Transforms execution traces into optimized computation graphs through:

1. Identity-based reference tracking
2. Value signature analysis
3. State mutation detection
4. Execution order constraints

Provides deterministic graph construction with parallelization opportunity detection.

Example:
    ```python
    from ember.xcs.tracer.autograph import AutoGraphBuilder
    from ember.xcs.tracer.xcs_tracing import TraceRecord
    from ember.xcs.engine import execute_graph

    # Execution traces from previous operator calls
    records = [
        TraceRecord(
            operator_name="TextProcessor",
            node_id="1",
            inputs={"text": "input text"},
            outputs={"tokens": ["input", "text"]}
        ),
        TraceRecord(
            operator_name="Embedding",
            node_id="2",
            inputs={"tokens": ["input", "text"]},
            outputs={"vectors": [[0.1, 0.2], [0.3, 0.4]]}
        ),
        TraceRecord(
            operator_name="Classifier",
            node_id="3",
            inputs={"vectors": [[0.1, 0.2], [0.3, 0.4]]},
            outputs={"class": "category_a", "confidence": 0.92}
        )
    ]

    # Construct optimized graph with parallelization metadata
    builder = AutoGraphBuilder()
    graph = builder.build_graph(records)

    # Execute with new inputs
    results = execute_graph(
        graph,
        inputs={"text": "different input"}
    )
    # Results contain final classification output
    ```
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.xcs_tracing import TraceRecord

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Dependency types between operator invocations."""

    DATA_FLOW = auto()  # Value passed directly between operators
    EXECUTION_ORDER = auto()  # Sequential execution requirement from control flow
    STATE_MUTATION = auto()  # Operators sharing stateful instance
    INFERRED = auto()  # Dependency determined through heuristic analysis


@dataclass
class DataReference:
    """Reference to data passed between operators.

    Tracks object identity and content signatures for dependency detection.
    Provides dual matching strategies: direct object identity for reference types
    and content-based signatures for value types.

    Attributes:
        obj_id: Object identity hash or None for primitive values
        path: Data location path (e.g., "outputs.result")
        signature: Content-based signature for value comparison
        producer: Node ID that produced this data
    """

    obj_id: Optional[int]
    path: str
    signature: str
    producer: Optional[str] = None


@dataclass
class DependencyNode:
    """Represents a single operator invocation in the dependency graph.

    Tracks data flow between operators, capturing inputs consumed and outputs
    produced. Functions as the primary data structure for dependency analysis.

    Attributes:
        trace_record: Source execution trace
        node_id: Unique node identifier
        inputs: Input data references keyed by name
        outputs: Output data references keyed by name
        dependencies: Incoming dependencies with dependency types
        outbound_edges: IDs of nodes that consume this node's outputs
    """

    trace_record: TraceRecord
    node_id: str
    inputs: Dict[str, DataReference] = field(default_factory=dict)
    outputs: Dict[str, DataReference] = field(default_factory=dict)
    dependencies: Dict[str, DependencyType] = field(default_factory=dict)
    outbound_edges: Set[str] = field(default_factory=set)


class DependencyAnalyzer:
    """Analyzes dependencies between operator executions.

    Performs multi-phase dependency analysis to identify relationships between
    operators. Employs several detection strategies in priority order:
    1. Object identity matching (most reliable)
    2. Content signature matching (for value types)
    3. Execution ordering constraints (for stateful operations)
    4. State mutation dependencies (for shared operator instances)

    The implementation prioritizes deterministic analysis over heuristics.
    """

    def __init__(self) -> None:
        """Initialize the dependency analyzer."""
        self.nodes: Dict[str, DependencyNode] = {}
        self.data_registry: Dict[Union[int, str], str] = {}
        self.operator_to_nodes: Dict[int, List[str]] = {}

    def analyze(self, records: List[TraceRecord]) -> Dict[str, DependencyNode]:
        """Performs multi-phase dependency analysis on trace records.

        Args:
            records: Execution trace records to analyze

        Returns:
            Mapping from node IDs to fully constructed dependency nodes

        Process:
        1. Creates dependency nodes for each trace record
        2. Extracts data references from inputs/outputs with path tracking
        3. Matches data flows using identity and signature comparison
        4. Applies execution ordering constraints for sequential operations
        5. Detects stateful operations requiring ordered execution
        6. Validates and optimizes the graph by removing redundant edges
        """
        if not records:
            return {}

        # Reset state for this analysis
        self.nodes.clear()
        self.data_registry.clear()
        self.operator_to_nodes.clear()

        # Create nodes and register operators
        for record in records:
            node_id = record.node_id
            node = DependencyNode(trace_record=record, node_id=node_id)
            self.nodes[node_id] = node

            # Track which nodes use the same operator instance
            op_id = id(record.operator) if record.operator else None
            if op_id:
                self.operator_to_nodes.setdefault(op_id, []).append(node_id)

        # Phase 1: Extract data references
        for _, node in self.nodes.items():
            self._extract_data_references(node)

        # Phase 2: Match data flow based on identity and signatures
        self._analyze_data_flow()

        # Phase 3: Apply execution ordering constraints
        self._apply_execution_ordering(records)

        # Phase 4: Detect stateful operation patterns
        self._detect_stateful_operations()

        # Phase 5: Validate and optimize dependency graph
        self._validate_and_optimize()

        return self.nodes

    def _extract_data_references(self, node: DependencyNode) -> None:
        """Extract data references from inputs and outputs.

        Creates DataReference objects for all input and output values,
        handling both dictionary and non-dictionary data structures uniformly.

        Args:
            node: Node to extract references for
        """
        # Extract input references
        self._extract_references_from_dict(
            data_dict=node.trace_record.inputs,
            references_dict=node.inputs,
            path_prefix="inputs",
        )

        # Extract output references - handle all types uniformly
        self._extract_references_from_dict(
            data_dict=node.trace_record.outputs,
            references_dict=node.outputs,
            path_prefix="outputs",
        )

        # Register outputs in the data registry for dependency tracking
        for _, ref in node.outputs.items():
            # Register by object ID for identity-based matching (most reliable)
            if ref.obj_id:
                self.data_registry[ref.obj_id] = node.node_id

            # Register by signature for content-based matching (fallback)
            if ref.signature:
                self.data_registry[ref.signature] = node.node_id

            # Set producer reference
            ref.producer = node.node_id

    def _extract_data_from_value(self, value: Any) -> Dict[str, Any]:
        """Extract dictionary data from various value types.

        Handles different types of input values and converts them to a
        standard dictionary format for dependency analysis.

        Args:
            value: Input value to extract data from

        Returns:
            Dictionary representation of the value

        Raises:
            ValueError: If value cannot be converted to a dictionary
        """
        # Handle None
        if value is None:
            return {}

        # Handle native dictionaries
        if isinstance(value, dict):
            return value

        # Handle Pydantic models (v2 and v1)
        if hasattr(value, "model_dump"):
            # Pydantic v2
            return value.model_dump()
        elif hasattr(value, "dict") and callable(value.dict):
            # Pydantic v1
            return value.dict()

        # Handle other mapping types
        if hasattr(value, "items") and callable(value.items):
            try:
                return dict(value.items())
            except (TypeError, ValueError):
                pass

        # Value is not a dictionary-like object
        # Just return a simple identifier key
        return {"value": value}

    def _extract_references_from_dict(
        self,
        data_dict: Dict[str, Any],
        references_dict: Dict[str, DataReference],
        path_prefix: str,
    ) -> None:
        """Extract data references from a dictionary.

        Args:
            data_dict: Dictionary to extract references from
            references_dict: Dictionary to store extracted references
            path_prefix: Path prefix for generated references
        """
        if not data_dict:
            return

        # Convert to standard dictionary format
        try:
            extracted_data = self._extract_data_from_value(data_dict)
        except ValueError:
            # If conversion fails, treat as a single value
            ref = self._create_reference(data_dict, path_prefix)
            references_dict["value"] = ref
            return

        # Process all keys in the dictionary
        for key, value in extracted_data.items():
            path = f"{path_prefix}.{key}"

            # Create reference for this value
            ref = self._create_reference(value, path)
            references_dict[key] = ref

            # Recursively process nested structures
            nested_dict = self._extract_data_from_value(value)
            if nested_dict and nested_dict != {"value": value}:
                # Only process if we got a meaningful dictionary
                nested_references: Dict[str, DataReference] = {}
                self._extract_references_from_dict(
                    data_dict=nested_dict,
                    references_dict=nested_references,
                    path_prefix=path,
                )
                # Add nested references with prefixed keys
                for nested_key, nested_ref in nested_references.items():
                    references_dict[f"{key}.{nested_key}"] = nested_ref

    def _create_reference(self, value: Any, path: str) -> DataReference:
        """Create a data reference for a value.

        Args:
            value: Value to create reference for
            path: Path to this value in the data structure

        Returns:
            Data reference for the value
        """
        # Use object identity for reference types, handle None specially
        obj_id = None
        if value is not None and not isinstance(value, (int, float, bool, str)):
            obj_id = id(value)

        # Create content signature
        signature = self._create_signature(value)

        return DataReference(obj_id=obj_id, path=path, signature=signature)

    def _create_signature(self, value: Any) -> str:
        """Create a content signature for a value.

        Optimized for speed and collision resistance.

        Args:
            value: Value to create signature for

        Returns:
            Content signature string
        """
        if value is None:
            return "none"

        # Handle primitive types directly
        if isinstance(value, (int, float, bool)):
            return f"{type(value).__name__}:{value}"

        if isinstance(value, str):
            # For strings, use a hash function to avoid excessive memory use
            if len(value) > 100:
                return f"str:{hashlib.md5(value.encode('utf-8')).hexdigest()}"
            return f"str:{value}"

        # Generate signature for complex types
        try:
            # Try a stable string representation first
            if hasattr(value, "__repr__"):
                repr_val = repr(value)
                # Avoid huge values
                if len(repr_val) > 1000:
                    repr_val = repr_val[:1000]
                # Create hash of the string representation
                md5_hash = hashlib.md5(repr_val.encode("utf-8")).hexdigest()
                return f"{type(value).__name__}:{md5_hash}"

            # Fall back to type and id
            return f"{type(value).__name__}:{id(value)}"
        except Exception:
            # Ultimate fallback
            return f"obj:{id(value)}"

    def _analyze_data_flow(self) -> None:
        """Analyze data flow between nodes based on references.

        Identifies where outputs from one node are used as inputs to another.
        Uses both identity-based matching and signature-based matching with
        priority given to identity for better reliability.
        """
        # Process nodes in order
        for consumer_id, consumer in self.nodes.items():
            # Check each input for a matching output
            for _, input_ref in consumer.inputs.items():
                producer_id = None

                # Try to find producer by object identity first
                if input_ref.obj_id and input_ref.obj_id in self.data_registry:
                    producer_id = self.data_registry[input_ref.obj_id]

                # Fall back to signature matching if no identity match
                elif input_ref.signature and input_ref.signature in self.data_registry:
                    producer_id = self.data_registry[input_ref.signature]

                # If we found a producer (that isn't the consumer itself)
                if producer_id and producer_id != consumer_id:
                    # A valid data dependency exists
                    consumer.dependencies[producer_id] = DependencyType.DATA_FLOW

                    # Register outbound edge on the producer
                    producer = self.nodes[producer_id]
                    producer.outbound_edges.add(consumer_id)

    def _apply_execution_ordering(self, records: List[TraceRecord]) -> None:
        """Apply execution ordering constraints.

        Ensures dependencies reflect the original execution order where necessary.

        Args:
            records: Original trace records in execution order
        """
        # Sort records by timestamp
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        ordered_node_ids = [r.node_id for r in sorted_records]

        # Track nodes that must maintain execution order
        order_dependent_nodes = set()

        # Find nodes that need execution order preservation
        for i in range(len(ordered_node_ids) - 1):
            current_id = ordered_node_ids[i]
            next_id = ordered_node_ids[i + 1]

            current_node = self.nodes[current_id]
            next_node = self.nodes[next_id]

            # Check if nodes share operator instances
            current_op = current_node.trace_record.operator
            next_op = next_node.trace_record.operator

            # Nodes using the same stateful operator likely need ordering
            if current_op is not None and current_op is next_op:
                order_dependent_nodes.add(current_id)
                order_dependent_nodes.add(next_id)

        # Apply ordering dependencies for order-dependent nodes
        for i in range(len(ordered_node_ids) - 1):
            current_id = ordered_node_ids[i]

            # Skip nodes not marked as order-dependent
            if current_id not in order_dependent_nodes:
                continue

            # Find the next order-dependent node
            for j in range(i + 1, len(ordered_node_ids)):
                next_id = ordered_node_ids[j]
                if next_id in order_dependent_nodes:
                    # Check if dependency already exists
                    curr_node = self.nodes[current_id]
                    next_node = self.nodes[next_id]
                    has_dependency = (
                        next_id in curr_node.outbound_edges
                        or current_id in next_node.dependencies
                    )

                    # Add execution order dependency if not already connected
                    if not has_dependency:
                        # Add dependency with execution order type
                        next_node.dependencies[
                            current_id
                        ] = DependencyType.EXECUTION_ORDER
                        self.nodes[current_id].outbound_edges.add(next_id)
                    break

    def _detect_stateful_operations(self) -> None:
        """Detect stateful operation patterns.

        Identifies where operations on the same operator instance might have
        dependencies due to shared state.
        """
        # Process operator instances with multiple invocations
        for _, node_ids in self.operator_to_nodes.items():
            if len(node_ids) < 2:
                continue

            # Find temporal order of these nodes
            temporal_order = sorted(
                node_ids, key=lambda n: self.nodes[n].trace_record.timestamp
            )

            # Connect nodes in temporal order if not already connected
            for i in range(len(temporal_order) - 1):
                current_id = temporal_order[i]
                next_id = temporal_order[i + 1]

                current_node = self.nodes[current_id]
                next_node = self.nodes[next_id]

                # Check if already connected via data flow
                if (
                    next_id in current_node.outbound_edges
                    or current_id in next_node.dependencies
                ):
                    continue

                # Check for non-trivial state in the operator instance
                current_rec = self.nodes[current_id].trace_record
                has_state = self._check_for_operator_state(current_rec.operator)

                # If operator has state, add state mutation dependency
                if has_state:
                    next_node.dependencies[current_id] = DependencyType.STATE_MUTATION
                    current_node.outbound_edges.add(next_id)

    def _check_for_operator_state(self, operator: Any) -> bool:
        """Check if an operator has non-trivial state.

        Args:
            operator: Operator instance to check

        Returns:
            True if operator likely has mutable state
        """
        if operator is None:
            return False

        # Check for instance variables beyond standard ones
        if hasattr(operator, "__dict__"):
            # Ignore standard attributes and methods
            state_attrs = [
                attr
                for attr in operator.__dict__
                if not attr.startswith("_") and not callable(getattr(operator, attr))
            ]
            return len(state_attrs) > 0

        return False

    def _validate_and_optimize(self) -> None:
        """Validate and optimize the dependency graph.

        Checks for cycles and ensures the graph is optimized for execution.
        """
        # Detect and break cycles
        if self._detect_and_break_cycles():
            logger.warning("Cycles detected and broken in dependency graph")

        # Optimize the graph (e.g., transitive reduction)
        self._perform_transitive_reduction()

    def _detect_and_break_cycles(self) -> bool:
        """Detect and break cycles in the dependency graph.

        Returns:
            True if cycles were detected and broken
        """
        # Build adjacency list
        graph: Dict[str, Set[str]] = {
            node_id: set(node.dependencies.keys())
            for node_id, node in self.nodes.items()
        }

        # Track visited nodes for cycle detection
        visited = set()
        temp_visited = set()
        cycles_broken = False

        def visit(node_id: str) -> bool:
            """DFS visit with cycle detection.

            Args:
                node_id: Node to visit

            Returns:
                True if a cycle was detected and broken
            """
            nonlocal cycles_broken

            # Skip already processed nodes
            if node_id in visited:
                return False

            # Check for cycle
            if node_id in temp_visited:
                # Found a cycle - break it by removing this edge
                # We need to identify which edge to remove
                for dependent_id in temp_visited:
                    if (
                        dependent_id in self.nodes
                        and node_id in self.nodes[dependent_id].dependencies
                    ):
                        # Remove lowest priority dependency
                        dep_type = self.nodes[dependent_id].dependencies[node_id]

                        # Prefer to break inferred or execution order dependencies first
                        if dep_type in (
                            DependencyType.INFERRED,
                            DependencyType.EXECUTION_ORDER,
                        ):
                            self.nodes[dependent_id].dependencies.pop(node_id)
                            if dependent_id in self.nodes[node_id].outbound_edges:
                                self.nodes[node_id].outbound_edges.remove(dependent_id)
                            cycles_broken = True
                            return True

                # If no preferred edge found, remove this one
                if node_id in self.nodes[list(temp_visited)[-1]].dependencies:
                    self.nodes[list(temp_visited)[-1]].dependencies.pop(node_id)
                    cycles_broken = True
                    return True

            # Mark as being visited
            temp_visited.add(node_id)

            # Visit neighbors
            for neighbor in graph.get(node_id, set()):
                if visit(neighbor):
                    return True

            # Mark as fully visited
            temp_visited.remove(node_id)
            visited.add(node_id)
            return False

        # Visit all nodes
        for node_id in list(self.nodes.keys()):
            if node_id not in visited:
                if visit(node_id):
                    # Reset visited sets on cycle detection
                    visited.clear()
                    temp_visited.clear()

        return cycles_broken

    def _perform_transitive_reduction(self) -> None:
        """Perform transitive reduction on the dependency graph.

        Removes redundant edges that don't affect the partial order.
        """
        # Build reachability table using Floyd-Warshall algorithm
        n = len(self.nodes)
        node_ids = list(self.nodes.keys())
        node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

        # Initialize reachability matrix
        reachable = [[False] * n for _ in range(n)]

        # Set direct connections
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            for dep_id in node.dependencies:
                if dep_id in node_to_index:
                    j = node_to_index[dep_id]
                    reachable[i][j] = True

        # Compute transitive closure
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    reachable[i][j] = reachable[i][j] or (
                        reachable[i][k] and reachable[k][j]
                    )

        # Perform reduction
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]

            # Track dependencies to remove
            to_remove = []

            # Check each direct dependency
            for dep_id in node.dependencies:
                if dep_id not in node_to_index:
                    continue

                j = node_to_index[dep_id]

                # Check if this dependency is redundant
                # A->C is redundant if A->B and B->C for some B
                is_redundant = False
                for k in range(n):
                    if k != i and k != j and reachable[i][k] and reachable[k][j]:
                        is_redundant = True
                        break

                if is_redundant:
                    to_remove.append(dep_id)

            # Remove redundant dependencies
            for dep_id in to_remove:
                if dep_id in node.dependencies:
                    node.dependencies.pop(dep_id)

                # Also update outbound edges
                if node_id in self.nodes[dep_id].outbound_edges:
                    self.nodes[dep_id].outbound_edges.remove(node_id)


class AutoGraphBuilder:
    """Builds optimized XCS computation graphs from execution traces.

    Transforms trace records into executable graphs with proper dependency
    preservation. Performs graph optimization through:

    1. Accurate dependency identification
    2. Parallelization opportunity detection
    3. Transitive reduction for minimal edge count
    4. Execution wave identification for parallel scheduling
    5. Operator binding with fallback mechanisms

    The implementation focuses on correctness and deterministic behavior
    first, with performance optimizations applied where safe.
    """

    def __init__(self) -> None:
        """Initializes an AutoGraphBuilder instance."""
        # Initialize dependencies storage
        self.nodes: Dict[str, DependencyNode] = {}

    def _extract_field_mappings(
        self, dep_node: DependencyNode, source_node_id: str, target_node_id: str
    ) -> Dict[str, str]:
        """Extract field-level mappings between nodes from dependency analysis.

        Uses the references in dependency nodes to determine which output fields
        from the source node are used as input fields in the target node.

        Args:
            dep_node: Dependency node containing data references
            source_node_id: ID of the node producing output
            target_node_id: ID of the node consuming input

        Returns:
            Dictionary mapping output field names to input field names
        """
        field_mappings = {}

        # Find the producer node for each input reference
        for input_field, input_ref in dep_node.inputs.items():
            # Check if this input came from the source node
            producer_id = input_ref.producer
            if producer_id == source_node_id:
                # Find matching output reference by path or signature
                for output_field, output_ref in self.nodes[
                    source_node_id
                ].outputs.items():
                    if (input_ref.obj_id and input_ref.obj_id == output_ref.obj_id) or (
                        input_ref.signature
                        and input_ref.signature == output_ref.signature
                    ):
                        # Map output field to input field
                        field_mappings[output_field] = input_field
                        break

        return field_mappings

    def build_graph(self, records: List[TraceRecord] = None, **kwargs) -> XCSGraph:
        """Builds an executable XCS graph from trace records.

        Constructs a graph in multiple phases:
        1. Analyzes dependencies between operator executions
        2. Creates graph nodes with callable operator references
        3. Adds edges based on identified dependencies
        4. Adds optimization metadata for execution scheduling

        Args:
            records: Trace records from operator executions
            **kwargs: Alternative keyword passing for records

        Returns:
            Executable XCS graph with optimization metadata

        Raises:
            ValueError: If invalid or conflicting records are provided
        """
        # Support both positional and keyword args for different calling conventions
        if records is None and "records" in kwargs:
            records = kwargs["records"]

        # Handle empty case
        if not records:
            return XCSGraph()

        # Create new graph
        graph = XCSGraph()

        # Map from trace record node_id to graph node_id
        node_id_map: Dict[str, str] = {}

        # Analyze dependencies
        analyzer = DependencyAnalyzer()
        dep_nodes = analyzer.analyze(records)
        # Store for field mapping extraction
        self.nodes = dep_nodes

        # First pass: Add nodes to the graph
        for i, record in enumerate(records):
            # Create a stable, predictable node ID
            graph_node_id = f"{record.operator_name}_{i}"
            node_id_map[record.node_id] = graph_node_id
            record.graph_node_id = graph_node_id

            # Add node to graph with appropriate operator
            graph.add_node(
                operator=self._create_operator_callable(trace_record=record),
                node_id=graph_node_id,
                name=record.operator_name,
            )

        # Second pass: Add edges based on dependencies
        for node_id, dep_node in dep_nodes.items():
            # Skip nodes not in the map (should never happen)
            if node_id not in node_id_map:
                continue

            graph_node_id = node_id_map[node_id]

            # Add edges for all dependencies
            for dep_id, dep_type in dep_node.dependencies.items():
                # Skip if dependent node not in map
                if dep_id not in node_id_map:
                    continue

                dep_graph_id = node_id_map[dep_id]

                # Extract field-level dependencies for precise mapping
                field_mappings = self._extract_field_mappings(dep_node, dep_id, node_id)

                # Add edge with dependency type and field mappings
                graph.add_edge(
                    from_id=dep_graph_id,
                    to_id=graph_node_id,
                    field_mappings=field_mappings,
                )

                # Optionally add edge metadata
                if "dependencies" not in graph.metadata:
                    graph.metadata["dependencies"] = {}

                edge_key = f"{dep_graph_id}->{graph_node_id}"
                graph.metadata["dependencies"][edge_key] = dep_type.name

        # Add optimization metadata
        self._add_execution_metadata(graph, dep_nodes, node_id_map)

        return graph

    def _add_execution_metadata(
        self,
        graph: XCSGraph,
        dep_nodes: Dict[str, DependencyNode],
        node_id_map: Dict[str, str],
    ) -> None:
        """Adds optimization metadata to the graph for efficient execution.

        Enhances the graph with execution hints for the scheduler:
        - Identifies parallelizable node groups
        - Detects aggregator nodes that combine parallel outputs
        - Organizes nodes into execution waves for wave-based scheduling
        - Identifies leaf nodes as potential outputs

        Args:
            graph: Target XCS graph to enhance with metadata
            dep_nodes: Analyzed dependency nodes
            node_id_map: Mapping from analysis node IDs to graph node IDs
        """
        # Add parallelization hints
        graph.metadata["parallelizable_nodes"] = []
        graph.metadata["parallel_groups"] = {}

        # Identify independent nodes (siblings without dependencies between them)
        parent_to_children: Dict[str, List[str]] = {}

        # Group nodes by parent
        for node_id, dep_node in dep_nodes.items():
            # Skip if not in the map
            if node_id not in node_id_map:
                continue

            graph_node_id = node_id_map[node_id]

            # Check dependencies to find parent
            dependencies = list(dep_node.dependencies.keys())

            if not dependencies:
                # Root node (no parent)
                parent_to_children.setdefault("root", []).append(graph_node_id)
            else:
                # Use first dependency as parent
                parent_id = dependencies[0]
                if parent_id in node_id_map:
                    parent_graph_id = node_id_map[parent_id]
                    parent_to_children.setdefault(parent_graph_id, []).append(
                        graph_node_id
                    )

        # Identify parallel groups - siblings with no dependencies between them
        group_id = 0
        for _, children in parent_to_children.items():
            if len(children) < 2:
                continue

            # Check if these children can run in parallel
            can_parallelize = True

            # Verify no child depends on another child
            for i, child1 in enumerate(children):
                for j, child2 in enumerate(children):
                    if i != j:
                        # Find original node IDs - extract from id maps
                        orig_id_map1 = [
                            (n, g) for n, g in node_id_map.items() if g == child1
                        ]
                        orig_id_map2 = [
                            (n, g) for n, g in node_id_map.items() if g == child2
                        ]

                        orig_id1 = orig_id_map1[0][0] if orig_id_map1 else None
                        orig_id2 = orig_id_map2[0][0] if orig_id_map2 else None

                        # Check for any dependencies between the nodes
                        has_dependency = False
                        if orig_id1 and orig_id2:
                            # Check if node2 depends on node1
                            if orig_id1 in dep_nodes:
                                node1 = dep_nodes[orig_id1]
                                if orig_id2 in node1.dependencies:
                                    has_dependency = True

                            # Check if node1 depends on node2
                            if not has_dependency and orig_id2 in dep_nodes:
                                node2 = dep_nodes[orig_id2]
                                if orig_id1 in node2.dependencies:
                                    has_dependency = True

                        if has_dependency:
                            can_parallelize = False
                            break

                if not can_parallelize:
                    break

            # If can parallelize, create a group
            if can_parallelize:
                group_name = f"parallel_group_{group_id}"
                group_id += 1

                # Add group to metadata
                graph.metadata["parallel_groups"][group_name] = children

                # Mark nodes as parallelizable
                graph.metadata["parallelizable_nodes"].extend(children)

        # Identify output node(s)
        leaf_nodes = []
        for node_id, node in graph.nodes.items():
            if not node.outbound_edges:
                leaf_nodes.append(node_id)

        if leaf_nodes:
            # Use the last leaf node as the primary output
            # Set both legacy and new-style metadata for compatibility
            graph.metadata["output_node"] = leaf_nodes[-1]
            graph.metadata["output_node_id"] = leaf_nodes[-1]
            # Store all leaf nodes for completeness
            graph.metadata["leaf_nodes"] = leaf_nodes

        # Add aggregator nodes identification
        graph.metadata["aggregator_nodes"] = []

        # Find aggregator nodes (nodes with multiple inputs)
        for node_id in graph.nodes:
            orig_id = next((n for n, g in node_id_map.items() if g == node_id), None)
            if orig_id and orig_id in dep_nodes:
                dep_node = dep_nodes[orig_id]
                if len(dep_node.dependencies) > 1:
                    # This node aggregates results from multiple sources
                    graph.metadata["aggregator_nodes"].append(node_id)

                    # Get mapped dependency IDs for this node
                    dep_graph_ids = []
                    for dep_id in dep_node.dependencies:
                        if dep_id in node_id_map:
                            dep_graph_ids.append(node_id_map[dep_id])

                    # Check each parallel group for membership
                    for group_name, members in graph.metadata[
                        "parallel_groups"
                    ].items():
                        # Check if multiple dependencies are from this group
                        common_members = [m for m in members if m in dep_graph_ids]
                        if len(common_members) > 1:
                            # Add information about what this node aggregates
                            if "aggregates_groups" not in graph.metadata:
                                graph.metadata["aggregates_groups"] = {}
                            graph.metadata["aggregates_groups"][node_id] = group_name

        # Add dependency waves for wave-based execution
        topo_order = graph.topological_sort()

        # Calculate in-degree for each node (number of dependencies)
        in_degree = {}
        for node_id in topo_order:
            in_degree[node_id] = len(graph.nodes[node_id].inbound_edges)
        waves = []
        remaining = set(topo_order)

        while remaining:
            wave = [node_id for node_id in remaining if in_degree[node_id] == 0]
            if not wave:
                break

            waves.append(wave)
            remaining.difference_update(wave)

            for node_id in wave:
                for out_edge in graph.nodes[node_id].outbound_edges:
                    if out_edge in in_degree:
                        in_degree[out_edge] -= 1

        graph.metadata["execution_waves"] = waves

    @staticmethod
    def _create_operator_callable(
        *, trace_record: TraceRecord
    ) -> Callable[[Dict[str, Any]], Any]:
        """Creates a callable that invokes the original operator with new inputs.

        Enables true JIT compilation rather than trace replay by:
        1. Retrieving operator reference through weak reference
        2. Executing the operator with provided inputs
        3. Falling back to recorded outputs if operator unavailable

        This approach balances memory safety with execution fidelity.

        Args:
            trace_record: Trace record with operator reference

        Returns:
            Function that executes the operator or replays traced outputs
        """
        import weakref

        # Store weak reference to avoid reference cycles
        operator_ref = (
            weakref.ref(trace_record.operator) if trace_record.operator else None
        )
        
        # Capture type information for reconstruction
        input_type_paths = trace_record.input_type_paths
        output_type_paths = trace_record.output_type_paths

        def operation_fn(*, inputs: Dict[str, Any]) -> Any:
            # Get the actual operator from the weak reference
            operator = operator_ref() if operator_ref else None
            
            if operator is None:
                # Operator no longer exists - reconstruct outputs with correct types
                if isinstance(trace_record.outputs, dict) and output_type_paths:
                    return AutoGraphBuilder._reconstruct_with_types(trace_record.outputs, output_type_paths)
                return trace_record.outputs

            # Execute with proper boundary crossing
            try:
                # Apply proper boundary crossing for inputs
                # Convert inputs to proper EmberModel types using the operator's specification
                typed_inputs = inputs
                if hasattr(operator, "specification") and hasattr(operator.specification, "validate_inputs"):
                    try:
                        typed_inputs = operator.specification.validate_inputs(inputs=inputs)
                    except Exception as e:
                        # Log but continue with original inputs if validation fails
                        import logging
                        logging.warning(f"Input validation failed during JIT execution: {e}")
                
                # Execute operator with validated inputs
                raw_output = operator(inputs=typed_inputs) if hasattr(operator, "__call__") else operator.forward(inputs=typed_inputs)
                
                # Apply proper boundary crossing for outputs
                # Ensure the output is a properly validated EmberModel
                if hasattr(operator, "specification") and hasattr(operator.specification, "validate_output"):
                    try:
                        return operator.specification.validate_output(output=raw_output)
                    except Exception as e:
                        # Log but continue with original output if validation fails
                        import logging
                        logging.warning(f"Output validation failed during JIT execution: {e}")
                        return raw_output
                        
                return raw_output
            except Exception as e:
                # Fallback to reconstructed outputs on error
                import logging
                logging.exception(f"Error during JIT execution: {e}")
                
                if isinstance(trace_record.outputs, dict) and output_type_paths:
                    return AutoGraphBuilder._reconstruct_with_types(trace_record.outputs, output_type_paths)
                return trace_record.outputs

        return operation_fn
        
    @staticmethod
    def _reconstruct_with_types(data: Dict[str, Any], type_paths: Dict[str, str]) -> Dict[str, Any]:
        """Reconstruct dictionary values with their proper types.

        Args:
            data: Dictionary with values to potentially reconstruct
            type_paths: Mapping of keys to type paths for reconstruction

        Returns:
            Dictionary with values reconstructed to proper types where possible
        """
        result = {}
        
        for key, value in data.items():
            if key in type_paths and isinstance(value, dict):
                type_path = type_paths[key]
                try:
                    # Import class
                    last_dot = type_path.rfind('.')
                    if last_dot > 0:
                        module_name = type_path[:last_dot]
                        class_name = type_path[last_dot+1:]
                        
                        module = __import__(module_name, fromlist=[class_name])
                        cls = getattr(module, class_name)
                        
                        # Reconstruct proper type
                        if hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
                            result[key] = cls.from_dict(value)
                            continue
                except (ImportError, AttributeError):
                    pass
            
            # Fallback to original value
            result[key] = value
        
        return result


def autograph(records=None):
    """Creates an XCS graph from execution trace records.
    
    A convenience function that constructs an AutoGraphBuilder and builds
    a graph from the provided trace records.
    
    Args:
        records: List of trace records from operator executions
        
    Returns:
        An XCS graph with nodes and edges based on the trace records
    """
    builder = AutoGraphBuilder()
    return builder.build_graph(records=records)
