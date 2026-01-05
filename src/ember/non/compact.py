"""Compact graph notation parsing and graph construction utilities.

This module implements the compact NON (Network of Networks) graph notation
that represents Ember graph architectures as short, evolvable strings.  The
notation supports both the new ``countType@provider/model(param=value)`` style
and the legacy ``count:TYPE:provider:model:temperature`` format.

The parser produces immutable dataclasses that capture nodes, optional flow
constraints, and the original textual representation.  Callers can inspect the
structures, render summaries, or convert graphs into execution plans.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TypeAlias

from .registry import DEFAULT_NODE_REGISTRY, NodeRegistry, NodeRole, NodeType

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSON: TypeAlias = JSONPrimitive | Dict[str, "JSON"] | List["JSON"]

# ---------------------------------------------------------------------------
# Dataclasses representing parsed compact graphs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompactNode:
    """Single node described by the compact NON notation."""

    id: str
    alias: Optional[str]
    type_code: str
    count: int
    model: Optional[str]
    params: Mapping[str, JSON] = field(default_factory=dict)
    original: str = ""

    @property
    def model_id(self) -> Optional[str]:
        """Return the fully qualified model identifier when present."""

        if not self.model:
            return None
        return self.model

    def replicated_models(self) -> List[str]:
        """Return the list of model identifiers for each replica."""

        if not self.model:
            return []
        return [self.model for _ in range(self.count)]


@dataclass(frozen=True, slots=True)
class FlowEndpoint:
    """Source or sink endpoint used in flow wiring."""

    node: str
    port: Optional[str] = None
    modifier: Optional[str] = None


@dataclass(frozen=True, slots=True)
class FlowSpec:
    """Edge set connecting sources to sinks."""

    sources: Tuple[FlowEndpoint, ...]
    targets: Tuple[FlowEndpoint, ...]
    original: str = ""


@dataclass(frozen=True, slots=True)
class CandidateStage:
    """Ensemble candidate stage."""

    node: CompactNode


@dataclass(frozen=True, slots=True)
class AggregatorStage:
    """Aggregation stage (e.g., majority vote)."""

    node: CompactNode


@dataclass(frozen=True, slots=True)
class JudgeStage:
    """Judge synthesis stage."""

    node: CompactNode


@dataclass(frozen=True, slots=True)
class VerifierStage:
    """Verifier stage."""

    node: CompactNode


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Structured view of the compact graph stages."""

    candidates: Tuple[CandidateStage, ...]
    aggregator: Optional[AggregatorStage]
    judge: Optional[JudgeStage]
    verifier: Optional[VerifierStage]


@dataclass(frozen=True, slots=True)
class CompactGraph:
    """Parsed compact graph consisting of nodes and optional flows."""

    nodes: Tuple[CompactNode, ...]
    node_types: Tuple[NodeType, ...]
    flows: Tuple[FlowSpec, ...]
    original_spec: Tuple[str, ...] = field(default_factory=tuple)
    expanded_components: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def sequential_order(self) -> Tuple[str, ...]:
        """Return nodes in declaration order for sequential execution."""

        return tuple(node.id for node in self.nodes)

    def to_execution_plan(self) -> ExecutionPlan:
        """Classify nodes into execution stages (candidates, judge, etc.)."""

        candidates: List[CandidateStage] = []
        aggregator: Optional[AggregatorStage] = None
        judge: Optional[JudgeStage] = None
        verifier: Optional[VerifierStage] = None

        for node, node_type in zip(self.nodes, self.node_types, strict=True):
            if node_type.role == NodeRole.ENSEMBLE:
                candidates.append(CandidateStage(node=node))
            elif node_type.role == NodeRole.AGGREGATOR:
                if aggregator is not None:
                    raise ValueError("Multiple aggregation stages detected; only one is supported.")
                aggregator = AggregatorStage(node=node)
            elif node_type.role == NodeRole.JUDGE:
                if judge is not None:
                    raise ValueError("Multiple judge stages detected; only one is supported.")
                judge = JudgeStage(node=node)
            elif node_type.role == NodeRole.VERIFIER:
                if verifier is not None:
                    raise ValueError("Multiple verifier stages detected; only one is supported.")
                verifier = VerifierStage(node=node)

        return ExecutionPlan(
            candidates=tuple(candidates),
            aggregator=aggregator,
            judge=judge,
            verifier=verifier,
        )


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------


_NEW_SYNTAX_RE = re.compile(
    r"^\s*"
    r"(?:(?P<count>\d+))?"
    r"(?P<code>[A-Z][A-Z0-9]*)"
    r"(?:!(?P<alias>[A-Za-z_][A-Za-z0-9_-]*))?"
    r"(?:@(?P<model>[^()\s]+))?"
    r"(?:\((?P<params>[^)]*)\))?"
    r"\s*$"
)


def parse_node_spec(spec: str, *, index: int) -> CompactNode:
    """Parse a single compact node specification string."""

    match = _NEW_SYNTAX_RE.match(spec)
    if not match:
        legacy = _parse_legacy_spec(spec, index=index)
        if legacy is not None:
            return legacy
        raise ValueError(f"Invalid compact node specification: '{spec}'")

    count_text = match.group("count")
    count = int(count_text) if count_text else 1
    type_code = match.group("code").upper()
    alias_raw = match.group("alias")
    alias = alias_raw.strip() if alias_raw else None
    model_raw = match.group("model")
    model = _normalize_model_identifier(model_raw) if model_raw else None
    params_text = match.group("params")
    params = _parse_params(params_text) if params_text else {}

    node_id = alias or _generate_node_id(type_code, index)
    return CompactNode(
        id=node_id,
        alias=alias,
        type_code=type_code,
        count=count,
        model=model,
        params=params,
        original=spec.strip(),
    )


def _parse_legacy_spec(spec: str, *, index: int) -> Optional[CompactNode]:
    if ":" not in spec:
        return None

    parts = [part.strip() for part in spec.split(":")]
    if len(parts) < 2:
        raise ValueError(f"Legacy NON spec must include at least count and type: '{spec}'")

    try:
        count = int(parts[0]) if parts[0] else 1
    except ValueError as exc:
        raise ValueError(f"Invalid count in legacy NON spec '{spec}': {parts[0]}") from exc

    type_code = (parts[1] or "E").upper()

    model: Optional[str] = None
    params: Dict[str, JSON] = {}

    if len(parts) >= 4:
        provider = parts[2]
        model_name = parts[3]
        if provider and model_name:
            model = _normalize_model_identifier(f"{provider}/{model_name}")
        elif provider:
            model = _normalize_model_identifier(provider)
    elif len(parts) == 3:
        provider = parts[2]
        model = _normalize_model_identifier(provider)

    if len(parts) >= 5 and parts[4]:
        params["temperature"] = _coerce_number(parts[4])

    node_id = _generate_node_id(type_code, index)
    return CompactNode(
        id=node_id,
        alias=None,
        type_code=type_code,
        count=count,
        model=model,
        params=params,
        original=spec.strip(),
    )


def _parse_params(params_text: str) -> Dict[str, JSON]:
    params: Dict[str, JSON] = {}
    if not params_text:
        return params

    for raw_item in _split_param_items(params_text.strip()):
        if not raw_item:
            continue
        if "=" not in raw_item:
            raise ValueError(f"Parameter must be key=value, got '{raw_item}'")
        key, value = raw_item.split("=", 1)
        normalized_key = _normalize_param_key(key.strip())
        params[normalized_key] = _parse_param_value(value.strip())
    return params


def _normalize_param_key(key: str) -> str:
    lowered = key.lower()
    if lowered in {"temp", "temperature"}:
        return "temperature"
    return key


def _parse_param_value(value: str) -> JSON:
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if (value.startswith("{") and value.endswith("}")) or (
        value.startswith("[") and value.endswith("]")
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _normalize_model_identifier(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    if "/" in model:
        return model
    if ":" in model:
        provider, rest = model.split(":", 1)
        return f"{provider}/{rest}"
    return model


def _coerce_number(value: str) -> JSON:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _generate_node_id(type_code: str, index: int) -> str:
    base = type_code.lower()
    return f"{base}{index + 1}"


def _split_param_items(text: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    depth = 0
    quote: Optional[str] = None
    escaped = False

    for char in text:
        if escaped:
            current.append(char)
            escaped = False
            continue

        if quote:
            current.append(char)
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue

        if char in "[{(":
            depth += 1
            current.append(char)
            continue

        if char in "]})":
            depth = max(0, depth - 1)
            current.append(char)
            continue

        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)

    return items


def parse_flow_spec(flow: str) -> FlowSpec:
    left, sep, right = flow.partition("->")
    if not sep:
        raise ValueError(f"Flow specification must contain '->': '{flow}'")
    sources = tuple(_parse_flow_endpoints(left))
    targets = tuple(_parse_flow_endpoints(right))
    if not sources or not targets:
        raise ValueError(f"Flow specification must include sources and targets: '{flow}'")
    return FlowSpec(sources=sources, targets=targets, original=flow.strip())


def _parse_flow_endpoints(text: str) -> List[FlowEndpoint]:
    entries = [entry.strip() for entry in text.split(",") if entry.strip()]
    endpoints: List[FlowEndpoint] = []
    for entry in entries:
        cleaned = entry.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1].strip()
        modifier = None
        if cleaned.endswith("[]"):
            modifier = "append"
            cleaned = cleaned[:-2]
        if cleaned.endswith(")") and "(" in cleaned:
            base, mod = cleaned.rsplit("(", 1)
            modifier = mod[:-1] if mod.endswith(")") else mod
            cleaned = base
        if "." in cleaned:
            node, port = cleaned.split(".", 1)
        else:
            node, port = cleaned, None
        endpoints.append(
            FlowEndpoint(
                node=node.strip(),
                port=port.strip() if port else None,
                modifier=modifier,
            )
        )
    return endpoints


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def build_graph(
    spec: Sequence[str] | Iterable[str] | str,
    *,
    components: Optional[Mapping[str, Sequence[str] | Iterable[str] | str]] = None,
    flows: Optional[Sequence[str] | Iterable[str] | str] = None,
    node_registry: Optional[NodeRegistry] = None,
) -> CompactGraph:
    """Expand components, parse nodes, and materialize a :class:`CompactGraph`.

    Args:
        spec: Sequence or newline/comma-delimited string of compact node
            specifications.
        components: Optional mapping of reusable component names to component
            specs (sequence or string).
        flows: Optional explicit flow declarations provided as a sequence or
            newline-delimited string.
        node_registry: Optional registry describing supported node types.  When
            omitted, the default registry that knows the core primitives is used.

    Returns:
        CompactGraph: Parsed graph with node metadata and flows.
    """

    normalized_spec = _normalize_spec_sequence(spec)
    normalized_components = {
        name: _normalize_spec_sequence(component_spec)
        for name, component_spec in (components or {}).items()
    }

    expanded_spec = tuple(_expand_spec_sequence(normalized_spec, normalized_components))
    parsed_nodes = [parse_node_spec(item, index=i) for i, item in enumerate(expanded_spec)]
    _validate_unique_node_ids(parsed_nodes)
    registry = node_registry or DEFAULT_NODE_REGISTRY
    node_types = tuple(_validate_nodes(parsed_nodes, registry=registry))
    nodes = tuple(parsed_nodes)

    if flows:
        normalized_flows = _normalize_flow_sequence(flows)
        parsed_flows = tuple(parse_flow_spec(flow) for flow in normalized_flows)
    else:
        parsed_flows = _default_sequential_flows(nodes)
    _validate_flow_references(nodes, parsed_flows)

    expanded_components = {name: tuple(values) for name, values in normalized_components.items()}

    return CompactGraph(
        nodes=nodes,
        node_types=node_types,
        flows=parsed_flows,
        original_spec=tuple(normalized_spec),
        expanded_components=expanded_components,
    )


def _normalize_spec_sequence(value: Sequence[str] | Iterable[str] | str) -> Tuple[str, ...]:
    return _normalize_sequence(value, label="spec", delimiters=(",", "\n", ";"))


def _normalize_flow_sequence(value: Sequence[str] | Iterable[str] | str) -> Tuple[str, ...]:
    return _normalize_sequence(value, label="flows", delimiters=("\n", ";"))


def _normalize_sequence(
    value: Sequence[str] | Iterable[str] | str,
    *,
    label: str,
    delimiters: Tuple[str, ...],
) -> Tuple[str, ...]:
    if isinstance(value, str):
        items = _split_items(value, delimiters)
        return tuple(item for item in items if item)

    try:
        iterator = iter(value)
    except TypeError as exc:  # pragma: no cover
        raise TypeError(f"{label} must be a string or iterable of strings") from exc

    normalized: List[str] = []
    for item in iterator:
        if not isinstance(item, str):
            raise TypeError(f"{label} entries must be strings, got {type(item).__name__}")
        cleaned = item.strip()
        if cleaned:
            normalized.append(cleaned)

    return tuple(normalized)


def _split_items(text: str, delimiters: Tuple[str, ...]) -> List[str]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    items: List[str] = []
    current: List[str] = []
    depth = 0
    quote: Optional[str] = None
    escaped = False

    delimiter_set = set(delimiters)

    for char in normalized_text:
        if escaped:
            current.append(char)
            escaped = False
            continue

        if quote:
            current.append(char)
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue

        if char in "([{":
            depth += 1
            current.append(char)
            continue

        if char in ")]}":
            depth = max(0, depth - 1)
            current.append(char)
            continue

        if char in delimiter_set and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)

    return items


def _expand_spec_sequence(
    spec: Sequence[str],
    components: Mapping[str, Sequence[str]],
    _stack: Optional[List[str]] = None,
) -> List[str]:
    stack = list(_stack or [])
    expanded: List[str] = []
    for item in spec:
        if item.startswith("$"):
            name = item[1:]
            if name in stack:
                cycle_path = " -> ".join(stack + [name])
                raise ValueError(f"Cyclic component reference detected: {cycle_path}")
            if name not in components:
                raise KeyError(f"Unknown component reference '${name}'")
            expanded.extend(_expand_spec_sequence(components[name], components, stack + [name]))
        else:
            expanded.append(item)
    return expanded


def _default_sequential_flows(nodes: Sequence[CompactNode]) -> Tuple[FlowSpec, ...]:
    flows: List[FlowSpec] = []
    for idx in range(len(nodes) - 1):
        left = nodes[idx].id
        right = nodes[idx + 1].id
        flow_str = f"{left} -> {right}"
        flows.append(
            FlowSpec(
                sources=(FlowEndpoint(node=left),),
                targets=(FlowEndpoint(node=right),),
                original=flow_str,
            )
        )
    return tuple(flows)


def _validate_unique_node_ids(nodes: Sequence[CompactNode]) -> None:
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for node in nodes:
        if node.id in seen:
            duplicates.append(node.id)
        else:
            seen[node.id] = node.original
    if duplicates:
        dup_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(
            "Duplicate NON node identifier(s) detected: "
            f"{dup_list}. Provide unique !alias values or remove duplicates."
        )


def _validate_flow_references(nodes: Sequence[CompactNode], flows: Sequence[FlowSpec]) -> None:
    if not flows:
        return
    valid_ids: Set[str] = {node.id for node in nodes}
    missing: Dict[str, List[str]] = {}
    for flow in flows:
        origin = flow.original or "<unnamed flow>"
        for endpoint in (*flow.sources, *flow.targets):
            if endpoint.node not in valid_ids:
                missing.setdefault(endpoint.node, []).append(origin)
    if missing:
        parts = []
        for node_id, origins in missing.items():
            origin_list = ", ".join(sorted(set(origins)))
            parts.append(f"{node_id} (referenced in: {origin_list})")
        raise ValueError("Flow references unknown node(s): " + "; ".join(parts))


def _validate_nodes(nodes: Sequence[CompactNode], *, registry: NodeRegistry) -> List[NodeType]:
    """Validate parsed nodes against the default registry."""

    node_types: List[NodeType] = []
    for node in nodes:
        node_type = registry.get(node.type_code)
        if node_type is None:
            known = ", ".join(registry.codes())
            raise ValueError(
                (
                    f"Unknown NON operator type '{node.type_code}' in '{node.original}'. "
                    f"Known types: {known}."
                )
            )

        if node_type.requires_model and not node.model:
            raise ValueError(
                (f"Node '{node.original}' of type {node.type_code} " "requires a model identifier.")
            )

        if not node_type.allows_model and node.model:
            raise ValueError(
                (
                    f"Node '{node.original}' of type {node.type_code} "
                    "does not accept a model binding."
                )
            )

        invalid_params = [key for key in node.params.keys() if not node_type.supports_param(key)]
        if invalid_params:
            allowed = (
                "any"
                if node_type.allowed_params is None
                else ", ".join(sorted(node_type.allowed_params))
            )
            raise ValueError(
                (
                    f"Node '{node.original}' of type {node.type_code} "
                    f"does not support parameter(s): {', '.join(sorted(invalid_params))}. "
                    f"Allowed parameters: {allowed}."
                )
            )

        node_types.append(node_type)

    return node_types


# ---------------------------------------------------------------------------
# Graph description helpers
# ---------------------------------------------------------------------------


def describe_graph(
    graph_id: str,
    spec: Sequence[str],
    *,
    components: Optional[Mapping[str, Sequence[str]]] = None,
    node_registry: Optional[NodeRegistry] = None,
) -> str:
    """Generate a human-readable summary for a compact graph specification."""

    graph = build_graph(spec, components=components, node_registry=node_registry)
    lines = [f"Graph '{graph_id}' ({len(graph.nodes)} stages):"]
    for node, node_type in zip(graph.nodes, graph.node_types, strict=True):
        details = []
        if node.count != 1:
            details.append(f"width={node.count}")
        if node.model_id:
            details.append(node.model_id)
        for key, value in node.params.items():
            details.append(f"{key}={value}")
        detail_str = ", ".join(details) if details else "no params"
        lines.append(f"  - {node.type_code} ({node_type.role.value}): {detail_str}")
    return "\n".join(lines)


__all__ = [
    "ExecutionPlan",
    "CandidateStage",
    "AggregatorStage",
    "JudgeStage",
    "VerifierStage",
    "CompactGraph",
    "CompactNode",
    "FlowEndpoint",
    "FlowSpec",
    "build_graph",
    "describe_graph",
    "parse_node_spec",
    "parse_flow_spec",
]
