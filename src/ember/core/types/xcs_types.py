"""
Type definitions for XCS (eXecution Control System) components.

This module provides type-safe definitions for XCS components,
including Graph, Plan, Task, and other execution-related types.
"""

from typing import Any, Dict, List, Protocol, TypeVar, Union, runtime_checkable, Callable
from typing_extensions import TypedDict, NotRequired

# Type variable for node input/output types
NodeInputT = TypeVar('NodeInputT')
NodeOutputT = TypeVar('NodeOutputT')


class XCSNodeAttributes(TypedDict, total=False):
    """
    Attributes that can be attached to a node in an XCS graph.
    """
    name: NotRequired[str]
    description: NotRequired[str]
    tags: NotRequired[List[str]]
    metadata: NotRequired[Dict[str, Any]]


class XCSNodeResult(TypedDict, total=False):
    """
    Result of executing a node in an XCS graph.
    """
    success: bool
    result: Any
    error: NotRequired[str]
    execution_time: NotRequired[float]
    metadata: NotRequired[Dict[str, Any]]


@runtime_checkable
class XCSNode(Protocol):
    """
    Protocol defining the interface for an XCS graph node.
    """
    node_id: str
    operator: Callable[..., Any]
    inbound_edges: List[str]
    outbound_edges: List[str]
    attributes: Dict[str, Any]
    captured_outputs: Any

    
@runtime_checkable
class XCSGraph(Protocol):
    """
    Protocol defining the interface for an XCS graph.
    """
    nodes: Dict[str, XCSNode]
    
    def add_node(self, node_id: str, operator: Callable[..., Any], **attributes: Any) -> None:
        """
        Add a node to the graph.
        """
        ...
        
    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add an edge between nodes.
        """
        ...
        
    def get_node(self, node_id: str) -> XCSNode:
        """
        Get a node by ID.
        """
        ...


@runtime_checkable
class XCSPlan(Protocol):
    """
    Protocol defining the interface for an XCS execution plan.
    """
    tasks: Dict[str, Any]  # XCSPlanTask
    original_graph: XCSGraph