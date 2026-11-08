"""What one jit call started: its model calls and its pending results.

A `Frame` belongs to one `jit` call, or to a `map` call made outside any jit.
Everything started inside it is recorded in it and in every enclosing frame:
the `Call` behind each dispatched request or stream, for the graph; the
`Pending` future of each request, so the boundary can wait for it and raise a
failure nobody read; and the completion of each stream's worker, so the
boundary can wait for it. A stream's failure belongs to its reader and
is never raised at the boundary.
"""

import time
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar

from ember.xcs.graph import Graph, Node

__all__ = ["Call", "CallKey", "Frame", "Pending", "Plan"]

T = TypeVar("T")

CallKey: TypeAlias = tuple[tuple[int, ...], int]
"""Where a call sits in the program: the spawn path of the greenlet that made
it and how many calls that greenlet had made before. A deterministic program
gives each call the same key on every run, whatever the timing."""


class Call:
    """One dispatched request, as the scheduler saw it.

    Attributes:
        label: What the request was dispatched as, usually the model id.
        deps: Calls whose results the dispatching greenlet had read.
        key: The call's place in the program.
        dispatched_at: `time.monotonic()` when it was dispatched.
        finished_at: `time.monotonic()` when its result arrived, set by the
            worker that ran it.
    """

    __slots__ = ("deps", "dispatched_at", "finished_at", "key", "label")

    def __init__(self, label: str, deps: frozenset["Call"], key: CallKey) -> None:
        self.label = label
        self.deps = deps
        self.key = key
        self.dispatched_at = time.monotonic()
        self.finished_at: float | None = None


class Pending(Future[T]):
    """The future of a call dispatched inside a scope.

    Attributes:
        call: The call; reading the outcome makes the reader's later calls
            depend on it.
        read: Whether any code has read the outcome. A failure that is never
            read is raised at the boundary of the jit call that created it.
    """

    def __init__(self, call: Call) -> None:
        super().__init__()
        self.call = call
        self.read = False


class Plan:
    """The calls of a previous run, keyed by their place in the program.

    Args:
        entries: For each call key, the call's label and its duration plus the
            longest chain of durations after it in that run.
    """

    def __init__(self, entries: Mapping[CallKey, tuple[str, float]]) -> None:
        self._entries = MappingProxyType(dict(entries))

    def expects(self, key: CallKey, label: str) -> bool:
        """Whether the previous run made a call with this label at ``key``."""
        entry = self._entries.get(key)
        return entry is not None and entry[0] == label

    def remaining_s(self, key: CallKey) -> float:
        """Predicted seconds from dispatching the call at ``key`` to the end of its chain."""
        entry = self._entries.get(key)
        return 0.0 if entry is None else entry[1]


class Frame:
    """The calls and pending results of one jit call.

    Args:
        parent: The enclosing frame, or ``None`` for the outermost one.
        plan: The previous run's plan; only an outermost frame uses one.
    """

    def __init__(self, parent: "Frame | None", plan: Plan | None) -> None:
        self.parent = parent
        self.plan = plan
        self.started_at = time.monotonic()
        self.calls: list[Call] = []
        # Results of every type the frame's code produced: user data.
        self.pending: list[Pending[Any]] = []
        # Completed by each stream's worker when it finishes.
        self.streams: list[Future[None]] = []
        self.settled = False
        """Whether the boundary waited for every pending result; only then is
        there a complete graph to record."""

    def chain(self) -> Iterator["Frame"]:
        """This frame and its enclosing frames, innermost first."""
        frame: Frame | None = self
        while frame is not None:
            yield frame
            frame = frame.parent

    def outermost(self) -> "Frame":
        """The frame that opened the scope."""
        *_, last = self.chain()
        return last

    def record(self, call: Call) -> None:
        """Adds ``call`` here and to every enclosing frame.

        The outermost frame drops its plan for the rest of the run as soon as
        the run departs from it.
        """
        for frame in self.chain():
            frame.calls.append(call)
        outermost = self.outermost()
        if outermost.plan is not None and not outermost.plan.expects(call.key, call.label):
            outermost.plan = None

    def track(self, pending: "Pending[T]") -> None:
        """Adds ``pending`` here and to every enclosing frame."""
        for frame in self.chain():
            frame.pending.append(pending)

    def track_stream(self, drained: "Future[None]") -> None:
        """Adds a stream's completion here and to every enclosing frame."""
        for frame in self.chain():
            frame.streams.append(drained)

    def planned_remaining_s(self, key: CallKey) -> float:
        """The plan's prediction for the call at ``key``; 0 without a plan."""
        plan = self.outermost().plan
        return 0.0 if plan is None else plan.remaining_s(key)

    def take_unread_failures(self) -> list[BaseException]:
        """Failures of pending results that nobody read, in creation order.

        Call only once every pending result is done. Marks them read, so an
        enclosing frame does not raise them again.
        """
        failures = []
        for pending in self.pending:
            if pending.read:
                continue
            error = pending.exception()
            if error is not None:
                pending.read = True
                failures.append(error)
        return failures

    def graph(self) -> Graph:
        """The calls made in this frame, as a `Graph`.

        Raises:
            RuntimeError: A call has not finished; graphs are built only after
                the frame's boundary waited for every call.
        """
        positions = {call: position for position, call in enumerate(self.calls)}
        nodes = []
        for position, call in enumerate(self.calls):
            if call.finished_at is None:
                raise RuntimeError(f"call {position} ({call.label}) has not finished")
            deps = sorted(positions[dep] for dep in call.deps if dep in positions)
            nodes.append(
                Node(
                    id=position,
                    label=call.label,
                    deps=tuple(deps),
                    start_s=call.dispatched_at - self.started_at,
                    end_s=call.finished_at - self.started_at,
                )
            )
        return Graph(tuple(nodes), wall_s=time.monotonic() - self.started_at)

    def next_plan(self, graph: Graph) -> Plan:
        """A plan for the next run, from this run's ``graph`` of this frame."""
        remaining = graph.remaining_s()
        return Plan({call.key: (call.label, remaining[i]) for i, call in enumerate(self.calls)})
