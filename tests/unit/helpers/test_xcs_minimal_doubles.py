"""Tests for minimal XCS test doubles."""

import pytest

from tests.helpers.xcs_minimal_doubles import (
    MinimalXCSGraph,
    MinimalXCSNode,
    minimal_autograph,
    minimal_jit,
    minimal_pmap,
    minimal_vmap,
)


class TestMinimalXCSNode:
    """Test the minimal XCS node implementation."""

    def test_node_creation(self):
        """Test basic node creation."""

        def add(x, y):
            return x + y

        node = MinimalXCSNode("test_node", add, 2, 3)
        assert node.node_id == "test_node"
        assert node.function == add
        assert node.args == (2, 3)
        assert node.kwargs == {}

    def test_node_execution(self):
        """Test node execution."""

        def multiply(x, y):
            return x * y

        node = MinimalXCSNode("mult_node", multiply, 4, 5)
        result = node.execute()
        assert result == 20

    def test_node_with_kwargs(self):
        """Test node with keyword arguments."""

        def power(base, exponent=2):
            return base**exponent

        node = MinimalXCSNode("pow_node", power, 3, exponent=3)
        result = node.execute()
        assert result == 27


class TestMinimalXCSGraph:
    """Test the minimal XCS graph implementation."""

    def test_graph_creation(self):
        """Test basic graph creation."""
        graph = MinimalXCSGraph()
        assert graph.nodes == {}
        assert graph.edges == {}

    def test_add_node_basic(self):
        """Test adding nodes with basic interface."""
        graph = MinimalXCSGraph()

        def square(x):
            return x * x

        node_id = graph.add_node("square", square, 5)
        assert node_id in graph.nodes
        assert graph.nodes[node_id].function == square

    def test_add_node_operator_interface(self):
        """Test adding nodes with operator interface."""
        graph = MinimalXCSGraph()

        def operator_func(**kwargs):
            return kwargs.get("value", 0) * 2

        node_id = graph.add_node(operator=operator_func, value=10)
        assert node_id in graph.nodes

    def test_add_edge(self):
        """Test adding edges between nodes."""
        graph = MinimalXCSGraph()

        node1 = graph.add_node("n1", lambda: 1)
        node2 = graph.add_node("n2", lambda: 2)

        graph.add_edge(node1, node2)
        assert node2 in graph.edges[node1]

    def test_execute_graph(self):
        """Test graph execution."""
        graph = MinimalXCSGraph()

        def add_one(x):
            return x + 1

        def double(x):
            return x * 2

        n1 = graph.add_node("add", add_one, 5)
        n2 = graph.add_node("double", double, 10)

        results = graph.execute([n1, n2])
        assert results[n1] == 6
        assert results[n2] == 20


class TestMinimalTransformations:
    """Test minimal transformation functions."""

    def test_minimal_vmap(self):
        """Test minimal vmap implementation."""

        def square(x):
            return x * x

        vmapped = minimal_vmap(square)
        result = vmapped([1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

    def test_minimal_pmap(self):
        """Test minimal pmap implementation."""

        def add_ten(x):
            return x + 10

        pmapped = minimal_pmap(add_ten)
        result = pmapped([1, 2, 3])
        assert result == [11, 12, 13]

    def test_minimal_jit(self):
        """Test minimal jit implementation."""

        def multiply(x, y):
            return x * y

        jitted = minimal_jit(multiply)

        # Should behave like original function
        assert jitted(3, 4) == 12

        # Should have graph attribute
        assert hasattr(jitted, "graph")
        assert hasattr(jitted, "get_graph")
        assert isinstance(jitted.get_graph(), MinimalXCSGraph)

    def test_minimal_autograph(self):
        """Test minimal autograph decorator."""

        @minimal_autograph
        def compute(x):
            return x**2

        # Should behave like original function
        assert compute(5) == 25

        # Should have graph attribute
        assert hasattr(compute, "graph")
        assert hasattr(compute, "get_graph")
        assert isinstance(compute.get_graph(), MinimalXCSGraph)


# Parameterized tests for better coverage
@pytest.mark.parametrize(
    "inputs,expected",
    [
        pytest.param([1, 2, 3], [1, 4, 9], id="simple-list"),
        pytest.param([0], [0], id="single-element"),
        pytest.param([], [], id="empty-list"),
    ],
)
def test_vmap_various_inputs(inputs, expected):
    """Test vmap with various input types."""

    def square(x):
        return x * x

    vmapped = minimal_vmap(square)
    assert vmapped(inputs) == expected


@pytest.mark.parametrize(
    "func,args,expected",
    [
        pytest.param(lambda x: x + 1, (5,), 6, id="increment"),
        pytest.param(lambda x, y: x * y, (3, 4), 12, id="multiply"),
        pytest.param(lambda: 42, (), 42, id="no-args"),
    ],
)
def test_node_various_functions(func, args, expected):
    """Test nodes with various function types."""
    node = MinimalXCSNode("test", func, *args)
    assert node.execute() == expected
