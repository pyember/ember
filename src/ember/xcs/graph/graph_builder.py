"""Graph builder for XCS graphs based on operator structure analysis.

Provides advanced graph building capabilities that analyze operator structure,
dependencies, and execution patterns to generate optimized execution graphs.
"""

import logging
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from ember.xcs.graph.xcs_graph import XCSGraph, XCSNode

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builder for computation graphs with node and edge creation."""   

    def build_from_operator(self, operator: Any) -> XCSGraph:
        """Build a graph from an operator's structure.
        
        Args:
            operator: Operator to build graph from
            
        Returns:
            Constructed computation graph
        """
        graph = XCSGraph()
        
        # Get operator attributes for graph building
        op_attrs = self._get_operator_attributes(operator)
        
        # Create nodes for main operator and nested operators
        root_node_id = graph.add_node(
            operator=operator,
            name=getattr(operator, "__name__", str(operator))
        )
        
        # Track as root node for execution
        graph.metadata["root_id"] = root_node_id
        
        # Return simple graph for now - enhancement will add more structure
        return graph
    
    def _get_operator_attributes(self, operator: Any) -> Dict[str, Any]:
        """Extract relevant attributes from an operator.
        
        Args:
            operator: Operator to analyze
            
        Returns:
            Dictionary of operator attributes
        """
        attrs = {}
        
        # For class instances, extract non-callable public attributes
        if not inspect.isfunction(operator) and not inspect.ismethod(operator):
            for attr_name in dir(operator):
                if attr_name.startswith('_'):
                    continue
                    
                attr = getattr(operator, attr_name, None)
                if callable(attr):
                    continue
                    
                attrs[attr_name] = attr
        
        return attrs


class EnhancedTraceGraphBuilder(GraphBuilder):
    """Advanced graph builder with dependency analysis.
    
    Analyzes execution traces and operator structure to build
    richer graphs with dependency tracking, enabling optimizations
    like automatic parallelization.    
    """
    
    def build_graph(self, records: List[Any]) -> XCSGraph:
        """Build a graph from trace records.
        
        Args:
            records: Trace records to build graph from
            
        Returns:
            Constructed computation graph
        """
        graph = XCSGraph()
        
        if not records:
            return graph
            
        # Extract operators and call relationships from records
        operators = {}
        call_edges = []
        
        for record in records:
            op_id = getattr(record, "operator_id", None)
            if op_id and op_id not in operators:
                operators[op_id] = getattr(record, "operator", None)
                
            caller_id = getattr(record, "caller_id", None)
            if op_id and caller_id:
                call_edges.append((caller_id, op_id))
        
        # Create nodes for operators
        node_map = {}
        for op_id, operator in operators.items():
            if not operator:
                continue
                
            node_id = graph.add_node(
                operator=operator,
                name=getattr(operator, "__name__", str(operator))
            )
            node_map[op_id] = node_id
        
        # Create edges for call relationships
        for caller_id, callee_id in call_edges:
            if caller_id in node_map and callee_id in node_map:
                graph.add_edge(node_map[caller_id], node_map[callee_id])
        
        return graph

    def build_from_trace(
        self, 
        operator: Any, 
        trace_data: Dict[str, Any], 
        recorded_calls: Dict[str, List[Dict[str, Any]]]
    ) -> XCSGraph:
        """Build a graph using execution trace data.
        
        Args:
            operator: Root operator
            trace_data: Trace context data
            recorded_calls: Recorded function calls during tracing
            
        Returns:
            Traced computation graph
        """
        graph = XCSGraph()
        
        # Create node for the root operator
        root_id = graph.add_node(
            operator=operator,
            name=getattr(operator, "__name__", str(operator))
        )
        
        # Add root ID to graph metadata
        graph.metadata["root_id"] = root_id
        
        # Process recorded calls to build edges
        call_nodes = {}
        
        for func_id, calls in recorded_calls.items():
            if not calls:
                continue
                
            # Create a node for this function
            func = trace_data.get("functions", {}).get(func_id)
            if func is None:
                continue
                
            node_id = graph.add_node(
                operator=func,
                name=getattr(func, "__name__", str(func))
            )
            
            call_nodes[func_id] = node_id
            
            # Add root-to-function edge if called directly by root
            if trace_data.get("caller_map", {}).get(func_id) == root_id:
                graph.add_edge(root_id, node_id)
        
        # Add edges between functions based on caller map
        for func_id, caller_id in trace_data.get("caller_map", {}).items():
            if func_id in call_nodes and caller_id in call_nodes:
                from_id = call_nodes[caller_id]
                to_id = call_nodes[func_id]
                graph.add_edge(from_id, to_id)
        
        return graph


class StructuralGraphBuilder(GraphBuilder):
    """Structure-based graph builder that analyzes operator composition patterns.
    
    Builds graphs by analyzing the structure of operators, their attributes, and
    their relationships to optimize execution plans without requiring execution tracing.
    """
    
    def __init__(self) -> None:
        """Initialize the structural graph builder."""
        self.recursive = True
    
    def set_recursive(self, recursive: bool) -> None:
        """Set whether to recursively analyze nested operators.
        
        Args:
            recursive: Whether to analyze recursively
        """
        self.recursive = recursive
    
    def build_graph(self, operator: Any) -> XCSGraph:
        """Build a graph by analyzing an operator's structure.
        
        Args:
            operator: The operator to analyze
            
        Returns:
            Constructed computational graph
        """
        graph = XCSGraph()
        
        # Create node for the root operator
        root_id = graph.add_node(
            operator=operator,
            name=getattr(operator, "__name__", str(operator))
        )
        
        # Add root ID to graph metadata
        graph.metadata["root_id"] = root_id
        
        # Recursively analyze the operator's structure and build the graph
        if self.recursive:
            self._analyze_structure(graph, operator, root_id, visited=set())
        
        return graph
    
    def _analyze_structure(
        self, graph: XCSGraph, obj: Any, parent_id: str, visited: Set[int], 
        attr_path: str = ""
    ) -> None:
        """Recursively analyze an object's structure and add nodes to the graph.
        
        Args:
            graph: The graph to build
            obj: The object to analyze
            parent_id: ID of the parent node
            visited: Set of object IDs that have already been visited
            attr_path: Attribute path from the root object
        """
        # Avoid cycles with visited set
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        
        # Extract attributes
        attrs = self._get_operator_attributes(obj)
        
        # Look for operator-like objects in attributes
        for attr_name, attr_val in attrs.items():
            # Skip non-object attributes
            if attr_val is None or isinstance(attr_val, (str, int, float, bool)):
                continue
                
            # Handle list/tuple of operators
            if isinstance(attr_val, (list, tuple)) and attr_val:
                for i, item in enumerate(attr_val):
                    # Check if item has forward method or is callable
                    if hasattr(item, "forward") or callable(item):
                        # Create a node for this item
                        node_name = f"{attr_name}[{i}]"
                        full_path = f"{attr_path}.{node_name}" if attr_path else node_name
                        
                        node_id = graph.add_node(
                            operator=item,
                            name=node_name
                        )
                        
                        # Add edge from parent to this node
                        graph.add_edge(parent_id, node_id)
                        
                        # Recursively analyze the item's structure
                        self._analyze_structure(graph, item, node_id, visited, full_path)
            
            # Handle individual operator-like objects
            elif hasattr(attr_val, "forward") or callable(attr_val):
                # Create a node for this attribute
                full_path = f"{attr_path}.{attr_name}" if attr_path else attr_name
                
                node_id = graph.add_node(
                    operator=attr_val,
                    name=attr_name
                )
                
                # Add edge from parent to this node
                graph.add_edge(parent_id, node_id)
                
                # Recursively analyze the attribute's structure
                self._analyze_structure(graph, attr_val, node_id, visited, full_path)