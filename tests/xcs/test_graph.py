"""Recorded graphs: what `jit(f).graph()` shows about the last run."""

from collections.abc import Iterator

import fakemodel
import pytest

from ember import xcs
from ember.xcs import Graph, Node, runtime

LATENCY_S = 0.05


def test_graph_is_none_before_the_first_call() -> None:
    assert xcs.jit(lambda: None).graph() is None


def test_a_recorded_graph_has_nodes_edges_timings_and_the_critical_path() -> None:
    model = fakemodel.FakeModel(LATENCY_S)

    @xcs.jit
    def program() -> list[str]:
        a, b = model("a"), model("b")
        c = model(a.text + b.text)
        d = model(c.text)
        e = model("e")
        return [d.text, e.text]

    program()
    graph = program.graph()
    assert graph is not None
    assert [node.label for node in graph.nodes] == ["fake/echo"] * 5
    assert graph.edges == ((0, 2), (1, 2), (2, 3), (2, 4))
    path = graph.critical_path
    assert path[0] in (0, 1) and path[1] == 2 and path[2] in (3, 4)
    assert all(edge in graph.edges for edge in zip(path, path[1:], strict=False))
    total = sum(graph.nodes[i].duration_s for i in path)
    assert total == pytest.approx(max(graph.remaining_s()))
    a, b, c, d, e = graph.nodes
    assert c.start_s >= max(a.end_s, b.end_s)
    assert d.start_s >= c.end_s
    for node in graph.nodes:
        assert 0 <= node.start_s <= node.end_s <= graph.wall_s
        assert node.duration_s >= LATENCY_S


def test_each_call_replaces_the_graph() -> None:
    model = fakemodel.FakeModel(0.0)

    @xcs.jit
    def program(n: int) -> list[str]:
        replies = [model(str(i)) for i in range(n)]
        return [reply.text for reply in replies]

    program(3)
    program(1)
    graph = program.graph()
    assert graph is not None
    assert len(graph.nodes) == 1


def test_map_items_depend_on_what_their_caller_read_and_hand_back_what_they_read() -> None:
    model = fakemodel.FakeModel(0.0)

    @xcs.jit
    def program() -> str:
        first = model("first").text
        items = xcs.map(lambda x: model(f"{first} {x}").text, [1, 2])
        return model(" ".join(items)).text

    program()
    graph = program.graph()
    assert graph is not None
    assert [node.deps for node in graph.nodes] == [(), (0,), (0,), (1, 2)]


def test_a_nested_jit_records_only_its_own_calls() -> None:
    model = fakemodel.FakeModel(0.0)

    @xcs.jit
    def inner(x: str) -> list[str]:
        first, second = model(x + "1"), model(x + "2")
        return [first.text, second.text]

    @xcs.jit
    def outer() -> list[str]:
        head = model("head").text
        return [head, *inner(head)]

    outer()
    outer_graph, inner_graph = outer.graph(), inner.graph()
    assert outer_graph is not None and inner_graph is not None
    assert len(outer_graph.nodes) == 3
    assert outer_graph.edges == ((0, 1), (0, 2))
    assert len(inner_graph.nodes) == 2
    assert inner_graph.edges == ()


def test_a_stream_is_a_node() -> None:
    def source() -> Iterator[str]:
        yield "token"

    @xcs.jit
    def program() -> list[str]:
        return list(runtime.dispatch_stream(source, label="fake/stream"))

    program()
    graph = program.graph()
    assert graph is not None
    assert [node.label for node in graph.nodes] == ["fake/stream"]


def test_remaining_time_and_critical_path_follow_the_longest_chain_of_durations() -> None:
    graph = Graph(
        nodes=(
            Node(id=0, label="short head", deps=(), start_s=0.0, end_s=1.0),
            Node(id=1, label="long head", deps=(), start_s=0.0, end_s=5.0),
            Node(id=2, label="short tail", deps=(0,), start_s=1.0, end_s=2.0),
            Node(id=3, label="long tail", deps=(1,), start_s=5.0, end_s=6.0),
        ),
        wall_s=6.0,
    )
    assert graph.remaining_s() == (2.0, 6.0, 1.0, 1.0)
    assert graph.critical_path == (1, 3)


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (Node(id=5, label="x", deps=(), start_s=0.0, end_s=1.0), "has id 5"),
        (Node(id=1, label="x", deps=(1,), start_s=0.0, end_s=1.0), "does not precede"),
        (Node(id=1, label="x", deps=(0, 0), start_s=0.0, end_s=1.0), "increasing and unique"),
        (Node(id=1, label="x", deps=(), start_s=1.0, end_s=0.5), "ends before it starts"),
    ],
)
def test_a_graph_rejects_inconsistent_nodes(node: Node, message: str) -> None:
    first = Node(id=0, label="x", deps=(), start_s=0.0, end_s=1.0)
    with pytest.raises(ValueError, match=message):
        Graph(nodes=(first, node), wall_s=1.0)
