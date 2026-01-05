"""Interfaces for provider-aware execution planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class ReservationHint:
    """Metadata describing a logical model invocation inside an IR node."""

    logical_model: str
    candidate_providers: tuple[str, ...] = ()
    concurrency_key: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reservation:
    """Result returned by a capacity broker before executing a node."""

    provider: str
    model: str
    credentials: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CapacityBroker(Protocol):
    """Resolve reservations for orchestration nodes."""

    def reserve(
        self,
        hint: ReservationHint,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[Reservation]:
        """Return a reservation or ``None`` when no capacity is available."""

    def release(
        self,
        reservation: Reservation,
        *,
        success: bool,
        usage: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Release a reservation, recording success and optional usage data."""


class NoOpCapacityBroker:
    """Default broker that always declines to reserve capacity."""

    def reserve(
        self,
        hint: ReservationHint,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[Reservation]:
        return None

    def release(
        self,
        reservation: Reservation,
        *,
        success: bool,
        usage: Optional[Mapping[str, Any]] = None,
    ) -> None:
        return None


NODE_METADATA_KEY = "ember_dispatch_hint"


_BROKER: CapacityBroker = NoOpCapacityBroker()


def get_capacity_broker() -> CapacityBroker:
    """Return the globally registered capacity broker."""

    return _BROKER


def set_capacity_broker(broker: Optional[CapacityBroker]) -> CapacityBroker:
    """Register a capacity broker, returning the active instance."""

    global _BROKER
    _BROKER = broker or NoOpCapacityBroker()
    return _BROKER


__all__ = [
    "CapacityBroker",
    "Reservation",
    "ReservationHint",
    "NoOpCapacityBroker",
    "NODE_METADATA_KEY",
    "get_capacity_broker",
    "set_capacity_broker",
]
