"""Metadata and validation registry for compact NON node types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Set


class NodeRole(str, Enum):
    """Semantic role of a node inside a compact NON graph."""

    ENSEMBLE = "ensemble"
    JUDGE = "judge"
    AGGREGATOR = "aggregator"
    VERIFIER = "verifier"
    OTHER = "other"


@dataclass(frozen=True)
class NodeType:
    """Definition of a compact NON node type."""

    code: str
    role: NodeRole
    description: str
    requires_model: bool = False
    allows_model: bool = True
    allowed_params: Optional[Set[str]] = None

    def supports_param(self, name: str) -> bool:
        if self.allowed_params is None:
            return True
        return name in self.allowed_params


class NodeRegistry:
    """Registry mapping node codes to metadata."""

    def __init__(self, mapping: Mapping[str, NodeType]):
        self._mapping: Dict[str, NodeType] = {
            code.upper(): node_type for code, node_type in mapping.items()
        }

    def get(self, code: str) -> Optional[NodeType]:
        return self._mapping.get(code.upper())

    def require(self, code: str) -> NodeType:
        node_type = self.get(code)
        if node_type is None:
            raise KeyError(code)
        return node_type

    def codes(self) -> Iterable[str]:
        return sorted(self._mapping.keys())

    def extend(self, mapping: Mapping[str, NodeType]) -> "NodeRegistry":
        """Return a new registry containing the supplied overrides.

        Args:
            mapping: Mapping of node code to :class:`NodeType` definitions.

        Returns:
            NodeRegistry: New registry with existing entries merged with
                ``mapping``.  Entries in ``mapping`` override existing codes.
        """

        combined: Dict[str, NodeType] = dict(self._mapping)
        for code, node_type in mapping.items():
            combined[code.upper()] = node_type
        return NodeRegistry(combined)


def _build_default_registry() -> NodeRegistry:
    common_params = {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "seed",
        "reasoning",
        "text",
        "max_output_tokens",
    }

    ensemble = NodeType(
        code="E",
        role=NodeRole.ENSEMBLE,
        description="Uniform ensemble of model replicas",
        requires_model=True,
        allowed_params=common_params,
    )

    judge = NodeType(
        code="J",
        role=NodeRole.JUDGE,
        description="Judge synthesis model",
        requires_model=True,
        allowed_params=common_params,
    )

    verifier = NodeType(
        code="V",
        role=NodeRole.VERIFIER,
        description="Verifier model stage",
        requires_model=True,
        allowed_params=common_params,
    )

    aggregator = NodeType(
        code="M",
        role=NodeRole.AGGREGATOR,
        description="Majority vote aggregation",
        requires_model=False,
        allows_model=False,
        allowed_params=None,
    )

    mapping: Dict[str, NodeType] = {
        "E": ensemble,
        "UE": ensemble,
        "J": judge,
        "JF": judge,
        "V": verifier,
        "M": aggregator,
        "MC": aggregator,
    }

    return NodeRegistry(mapping)


DEFAULT_NODE_REGISTRY = _build_default_registry()

__all__ = ["NodeRole", "NodeType", "NodeRegistry", "DEFAULT_NODE_REGISTRY"]
