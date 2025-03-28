"""
Minimal test doubles for XCS components.

This module provides simplified test doubles that implement just enough functionality
to test client code without duplicating the entire implementation. Following the
principle of "avoid overmocking" from CLAUDE.md guidelines.
"""

import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Type variables
T = TypeVar("T")
U = TypeVar("U")


class MinimalXCSNode:
    """Minimal test double for a graph node."""

    def __init__(self, node_id: str, function: Callable, *args: Any, **kwargs: Any):
        """Initialize with essential attributes."""
        self.node_id = node_id
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def execute(self) -> Any:
        """Execute the function with its arguments."""
        return self.function(*self.args, **self.kwargs)


class MinimalXCSGraph:
    """Minimal test double for XCS graph."""

    def __init__(self):
        """Initialize an empty graph."""
        self.nodes: Dict[str, MinimalXCSNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self._node_counter = 0
        self._results: Dict[str, Any] = {}

    def add_node(self, name=None, func=None, *args, **kwargs) -> str:
        """Add a node to the graph with minimal implementation."""
        # Handle the operator interface (used most often in tests)
        if "operator" in kwargs:
            operator = kwargs.pop("operator")
            node_id = kwargs.pop("node_id", f"node_{self._node_counter}")
            self._node_counter += 1

            # Create a simple wrapper function
            def wrapper_func(*args, **kw):
                return operator(**kw)

            # Store the node
            self.nodes[node_id] = MinimalXCSNode(node_id, wrapper_func)
            self.edges[node_id] = []
            return node_id

        # Handle the original interface
        else:
            if name is None or func is None:
                raise ValueError("Must provide name and func")

            node_id = f"{name}_{self._node_counter}"
            self._node_counter += 1

            # Create and store the node
            self.nodes[node_id] = MinimalXCSNode(node_id, func, *args, **kwargs)
            self.edges[node_id] = []

            return node_id

    def add_edge(self, source=None, target=None, **kwargs) -> None:
        """Add an edge between nodes with minimal implementation."""
        # Handle XCS interface
        if "from_id" in kwargs or "to_id" in kwargs:
            source = kwargs.get("from_id")
            target = kwargs.get("to_id")

        # Add the edge if nodes exist
        if source in self.nodes and target in self.nodes:
            self.edges[source].append(target)

    def execute(self, output_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute the graph with minimal functionality."""
        # If no output nodes specified, use all nodes
        if output_nodes is None:
            output_nodes = list(self.nodes.keys())

        # Reset results
        self._results = {}

        # Execute requested nodes
        for node_id in output_nodes:
            if node_id in self.nodes:
                self._results[node_id] = self.nodes[node_id].execute()

        # Return results for requested nodes
        return {
            node_id: self._results[node_id]
            for node_id in output_nodes
            if node_id in self._results
        }


def minimal_vmap(func: Callable[[T], U]) -> Callable[[List[T]], List[U]]:
    """Minimal implementation of vectorized mapping."""

    @functools.wraps(func)
    def vectorized_func(inputs):
        """Apply function to each input element."""
        if isinstance(inputs, list):
            return [func(x) for x in inputs]
        # Handle numpy arrays or other sequence types
        try:
            return [func(inputs[i]) for i in range(len(inputs))]
        except:
            # Fall back to just applying the function
            return func(inputs)

    return vectorized_func


def minimal_pmap(func: Callable[[T], U]) -> Callable[[List[T]], List[U]]:
    """Minimal implementation of parallel mapping.

    In test environments, we don't need actual parallelism,
    so this is just a sequential implementation.
    """

    @functools.wraps(func)
    def parallel_func(inputs):
        """Apply function to each input element."""
        if isinstance(inputs, list):
            return [func(x) for x in inputs]
        # Handle numpy arrays or other sequence types
        try:
            return [func(inputs[i]) for i in range(len(inputs))]
        except:
            # Fall back to just applying the function
            return func(inputs)

    return parallel_func


def minimal_jit(func):
    """Minimal implementation of JIT compilation.

    For testing, this is just the identity function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Add graph attribute expected by some tests
    wrapper.graph = MinimalXCSGraph()
    wrapper.get_graph = lambda: wrapper.graph

    return wrapper


class MinimalAutographDecorator:
    """Minimal implementation of autograph decorator."""

    def __init__(self):
        """Initialize with an empty graph."""
        self.graph = MinimalXCSGraph()

    def __call__(self, func):
        """Decorate a function to capture its graph."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add graph attribute expected by tests
        wrapper.graph = self.graph
        wrapper.get_graph = lambda: wrapper.graph

        return wrapper


# Create singleton instance
minimal_autograph = MinimalAutographDecorator()

# Export minimal test doubles
__all__ = [
    "MinimalXCSNode",
    "MinimalXCSGraph",
    "minimal_vmap",
    "minimal_pmap",
    "minimal_jit",
    "minimal_autograph",
]
