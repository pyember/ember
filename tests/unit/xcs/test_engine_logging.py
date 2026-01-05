"""Tests for execution engine observability hooks."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import pytest

from ember.xcs.compiler.graph import IRGraph, node_from_callable
from ember.xcs.runtime.broker import (
    NODE_METADATA_KEY,
    CapacityBroker,
    Reservation,
    ReservationHint,
)
from ember.xcs.runtime.engine import ExecutionEngine


class _FailingBroker(CapacityBroker):
    def reserve(
        self, hint: ReservationHint, *, timeout: Optional[float] = None
    ) -> Optional[Reservation]:
        raise RuntimeError("capacity lookup failed")

    def release(
        self,
        reservation: Reservation,
        *,
        success: bool,
        usage: Optional[Mapping[str, Any]] = None,
    ) -> None:
        return None


@pytest.fixture()
def engine() -> ExecutionEngine:
    inst = ExecutionEngine(max_workers=1, broker=_FailingBroker())
    try:
        yield inst
    finally:
        inst.shutdown()


def _build_graph() -> IRGraph:
    graph = IRGraph()

    def noop() -> int:
        return 7

    metadata = {
        "is_return": True,
        "call_spec_pos": tuple(),
        "call_spec_kw": {},
        "return_vars": ("out",),
    }
    metadata[NODE_METADATA_KEY] = ReservationHint(
        logical_model="demo-model",
        candidate_providers=("provider-a",),
        concurrency_key="demo",
        metadata={"priority": "high"},
    )
    node = node_from_callable("node_1", noop, (), ("out",), metadata)
    graph = graph.add_node(node)
    return graph


def test_broker_failure_logs_warning(
    engine: ExecutionEngine, caplog: pytest.LogCaptureFixture
) -> None:
    graph = _build_graph()
    with caplog.at_level(logging.WARNING):
        result = engine.execute(graph, tuple(), {})
    assert result == 7

    records = [record for record in caplog.records if "Capacity broker failed" in record.message]
    assert len(records) == 1
    record = records[0]
    assert record.node_id == "node_1"
    assert record.exc_class == "RuntimeError"
    hint = record.hint
    assert hint["logical_model"] == "demo-model"


def test_broker_warning_sampling(engine: ExecutionEngine, caplog: pytest.LogCaptureFixture) -> None:
    graph = _build_graph()
    with caplog.at_level(logging.WARNING):
        engine.execute(graph, tuple(), {})
        caplog.clear()
        engine.execute(graph, tuple(), {})
    records = [record for record in caplog.records if "Capacity broker failed" in record.message]
    assert not records
