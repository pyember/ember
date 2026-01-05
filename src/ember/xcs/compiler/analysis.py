"""Static analysis utilities for classifying function operations.

This module provides comprehensive operation classification for XCS, determining:
- Operation kind (tensor, orchestration, hybrid, neutral, unknown)
- JAX traceability (whether JAX transforms can be safely applied)
- Effect risk (I/O, network, subprocess operations that should not be traced)
- Confidence and evidence for debugging

The classification follows a priority system:
1. Explicit markers (@mark_tensor, @mark_orchestration, @mark_hybrid)
2. Closure/global binding analysis
3. Module-based detection
4. AST-based heuristics with local import resolution

Design Philosophy:
- Explicit over implicit: user markers always win
- Conservative on ambiguity: unknown != hybrid
- Module-aware: "torch.nn.Module" is tensor, not orchestration
- Effect-safe: never JAX-trace code that might have side effects
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import re
import textwrap
import threading
import weakref
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

try:  # JAX is an optional dependency for some Ember installations.
    import jax
    from jax import core as jax_core
    from jax import tree_util as jax_tree_util
except Exception:  # pragma: no cover - import guards for optional dependency
    jax = None  # type: ignore[assignment]
    jax_core = None  # type: ignore[assignment]
    jax_tree_util = None  # type: ignore[assignment]

try:  # ShapeDtypeStruct lives on the top-level jax module.
    from jax import ShapeDtypeStruct as _ShapeDtypeStruct
except Exception:  # pragma: no cover - keep tests isolated from import churn
    _ShapeDtypeStruct = ()  # type: ignore[assignment]

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


# -----------------------------------------------------------------------------
# Enums for rich classification
# -----------------------------------------------------------------------------


class OpKind(Enum):
    """Classification of operation type."""

    TENSOR = auto()  # Pure numerical/tensor operations
    ORCHESTRATION = auto()  # LLM/API/network operations
    HYBRID = auto()  # Mix of tensor and orchestration
    NEUTRAL = auto()  # Pure Python, no indicators either way
    UNKNOWN = auto()  # Inspection failed, cannot determine


class Traceability(Enum):
    """Whether operations can be traced by JAX."""

    JAX = auto()  # Safe for jax.jit/vmap/grad
    NON_JAX = auto()  # Numeric but not JAX-compatible (torch, numpy, tf)
    UNKNOWN = auto()  # Cannot determine


class EffectRisk(Enum):
    """Risk level for side effects during tracing."""

    LOW = auto()  # Pure computation, safe to trace speculatively
    MEDIUM = auto()  # May have effects but not critical (logging, etc.)
    HIGH = auto()  # Has I/O, network, subprocess - never JAX-trace


# -----------------------------------------------------------------------------
# Module and keyword configuration
# -----------------------------------------------------------------------------

# JAX-traceable modules - safe for jax.jit/vmap/grad
_JAX_TRACEABLE_PREFIXES: Tuple[str, ...] = (
    "jax.",
    "jaxlib.",
    "flax.",
    "equinox.",
    "optax.",
    "haiku.",
    "chex.",
    "rlax.",
    "dm_pix.",
)

# Numeric but NOT JAX-traceable - tensor ops but cannot use JAX transforms
_NUMERIC_NON_JAX_PREFIXES: Tuple[str, ...] = (
    "numpy.",
    "np.",
    "torch.",
    "tensorflow.",
    "tf.",
    "scipy.",
    "sklearn.",
    "pandas.",
    "cupy.",
)

# Combined tensor prefixes for kind classification
_TENSOR_MODULE_PREFIXES: Tuple[str, ...] = _JAX_TRACEABLE_PREFIXES + _NUMERIC_NON_JAX_PREFIXES

# Orchestration modules
_ORCHESTRATION_MODULE_PREFIXES: Tuple[str, ...] = (
    "anthropic.",
    "openai.",
    "langchain.",
    "litellm.",
    "cohere.",
    "google.generativeai.",
    "mistralai.",
    "replicate.",
    "huggingface_hub.",
    "transformers.",
    "ember.gateway.",
    "ember.operators.",
)

# Keywords that strongly indicate tensor operations
# These should be whole-token matches, not substrings
_TENSOR_KEYWORDS: FrozenSet[str] = frozenset(
    {
        "jax",
        "jnp",
        "numpy",
        "ndarray",
        "tensor",
        "matmul",
        "einsum",
        "lax",
        "vmap",
        "pmap",
        "torch",
        "tensorflow",
    }
)

# Keywords that strongly indicate orchestration
# Removed ambiguous terms: "model", "call", "api"
_ORCHESTRATION_KEYWORDS: FrozenSet[str] = frozenset(
    {
        "llm",
        "anthropic",
        "openai",
        "prompt",
        "completion",
        "chat",
        "mcp",
        "langchain",
        "litellm",
        "gemini",
        "cohere",
    }
)

# Effectful builtins that indicate HIGH effect risk
_EFFECTFUL_BUILTINS: FrozenSet[str] = frozenset(
    {
        "open",
        "print",
        "input",
        "exec",
        "eval",
        "compile",
        "__import__",
    }
)

# Effectful module prefixes
_EFFECTFUL_MODULE_PREFIXES: Tuple[str, ...] = (
    "os.",
    "pathlib.",
    "subprocess.",
    "socket.",
    "requests.",
    "httpx.",
    "aiohttp.",
    "urllib.",
    "http.",
    "asyncio.",
    "multiprocessing.",
    "threading.",
    "sqlite3.",
    "psycopg",
    "pymysql.",
    "redis.",
    "boto3.",
    "google.cloud.",
    "azure.",
)

# Token boundary pattern for keyword matching
_TOKEN_BOUNDARY = re.compile(r"[._\-\s]+")


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpDecision:
    """Rich decision object for operation classification.

    This replaces the simpler OperationSummary with a more complete model
    that separates kind from traceability and includes effect risk.
    """

    kind: OpKind
    jax_traceable: Traceability = Traceability.UNKNOWN
    effect_risk: EffectRisk = EffectRisk.LOW
    confidence: float = 0.0
    tensor_evidence: FrozenSet[str] = frozenset()
    orchestration_evidence: FrozenSet[str] = frozenset()
    effect_evidence: FrozenSet[str] = frozenset()
    reason: str = ""

    @property
    def is_tensor(self) -> bool:
        """True if kind is TENSOR."""
        return self.kind == OpKind.TENSOR

    @property
    def is_orchestration(self) -> bool:
        """True if kind is ORCHESTRATION."""
        return self.kind == OpKind.ORCHESTRATION

    @property
    def is_hybrid(self) -> bool:
        """True if kind is HYBRID."""
        return self.kind == OpKind.HYBRID

    @property
    def is_neutral(self) -> bool:
        """True if kind is NEUTRAL."""
        return self.kind == OpKind.NEUTRAL

    @property
    def is_unknown(self) -> bool:
        """True if kind is UNKNOWN."""
        return self.kind == OpKind.UNKNOWN

    @property
    def can_jax_trace(self) -> bool:
        """True if safe to apply JAX transforms."""
        return (
            self.jax_traceable == Traceability.JAX
            and self.effect_risk != EffectRisk.HIGH
        )

    @property
    def should_use_orchestration_path(self) -> bool:
        """True if orchestration execution path should be used."""
        return self.kind in (OpKind.ORCHESTRATION, OpKind.HYBRID, OpKind.UNKNOWN)


@dataclass(slots=True)
class OperationSummary:
    """Legacy summary for backward compatibility.

    This maintains API compatibility with existing code while internally
    using the richer OpDecision model.
    """

    tensor_ops: Set[str] = field(default_factory=set)
    orchestration_ops: Set[str] = field(default_factory=set)
    _decision: Optional[OpDecision] = field(default=None, repr=False)

    @property
    def has_tensor_ops(self) -> bool:
        return bool(self.tensor_ops)

    @property
    def has_orchestration_ops(self) -> bool:
        return bool(self.orchestration_ops)

    @property
    def only_tensor_ops(self) -> bool:
        return self.has_tensor_ops and not self.has_orchestration_ops

    @property
    def only_orchestration_ops(self) -> bool:
        return self.has_orchestration_ops and not self.has_tensor_ops

    @property
    def is_hybrid(self) -> bool:
        return self.has_tensor_ops and self.has_orchestration_ops

    @property
    def decision(self) -> OpDecision:
        """Access the underlying rich decision object."""
        if self._decision is not None:
            return self._decision
        # Synthesize from legacy fields
        kind = self._infer_kind()
        return OpDecision(
            kind=kind,
            tensor_evidence=frozenset(self.tensor_ops),
            orchestration_evidence=frozenset(self.orchestration_ops),
        )

    def _infer_kind(self) -> OpKind:
        """Infer OpKind from legacy tensor_ops/orchestration_ops sets."""
        has_tensor = bool(self.tensor_ops)
        has_orch = bool(self.orchestration_ops)
        has_unknown = "unknown" in self.tensor_ops or "unknown" in self.orchestration_ops

        if has_unknown:
            return OpKind.UNKNOWN
        if has_tensor and has_orch:
            return OpKind.HYBRID
        if has_tensor:
            return OpKind.TENSOR
        if has_orch:
            return OpKind.ORCHESTRATION
        return OpKind.NEUTRAL


# -----------------------------------------------------------------------------
# Analysis caching
# -----------------------------------------------------------------------------

_ANALYSIS_CACHE: "weakref.WeakKeyDictionary[Any, Tuple[OpDecision, int]]" = (
    weakref.WeakKeyDictionary()
)
_CACHE_LOCK = threading.Lock()


def _get_code_hash(func: Callable[..., Any]) -> int:
    """Compute a hash of function code for cache invalidation."""
    code = getattr(func, "__code__", None)
    if code is None:
        return id(func)
    # Hash bytecode and constants for change detection
    h = hashlib.md5(code.co_code, usedforsecurity=False)
    h.update(str(code.co_consts).encode())
    return int(h.hexdigest()[:16], 16)


# -----------------------------------------------------------------------------
# AST visitor with import alias tracking
# -----------------------------------------------------------------------------


class _OperationVisitor(ast.NodeVisitor):
    """AST visitor with local import alias resolution and boundary-aware matching.

    This visitor tracks:
    - Local imports and their aliases (e.g., `import jax.numpy as jnp`)
    - Function calls and attribute accesses
    - Effectful operations

    It uses module-boundary-aware matching to avoid false positives.
    """

    def __init__(self) -> None:
        self.tensor_ops: Set[str] = set()
        self.orchestration_ops: Set[str] = set()
        self.effect_ops: Set[str] = set()
        self.alias_map: Dict[str, str] = {}  # alias -> full module name
        self.is_async: bool = False

    def visit_Import(self, node: ast.Import) -> Any:
        """Track import statements for alias resolution."""
        for alias in node.names:
            name = alias.name
            as_name = alias.asname or name.split(".")[0]
            self.alias_map[as_name] = name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Track from-import statements for alias resolution."""
        module = node.module or ""
        for alias in node.names:
            name = alias.name
            as_name = alias.asname or name
            full_name = f"{module}.{name}" if module else name
            self.alias_map[as_name] = full_name
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Detect async functions as effectful."""
        self.is_async = True
        self.effect_ops.add("async_def")
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> Any:
        """Detect await expressions as effectful."""
        self.effect_ops.add("await")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Analyze function calls."""
        name = self._extract_name(node.func)
        resolved = self._resolve_alias(name)
        self._record_name(resolved)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Analyze attribute accesses."""
        name = self._extract_name(node)
        resolved = self._resolve_alias(name)
        self._record_name(resolved)
        self.generic_visit(node)

    def _resolve_alias(self, dotted_name: str) -> str:
        """Resolve import aliases to full module names."""
        if not dotted_name:
            return dotted_name

        parts = dotted_name.split(".")
        first = parts[0]

        if first in self.alias_map:
            resolved_first = self.alias_map[first]
            if len(parts) > 1:
                return f"{resolved_first}.{'.'.join(parts[1:])}"
            return resolved_first

        return dotted_name

    def _record_name(self, dotted_name: str) -> None:
        """Classify a dotted name using boundary-aware matching."""
        if not dotted_name:
            return

        lowered = dotted_name.lower()

        # Check for effectful operations first
        base_name = dotted_name.split(".")[0]
        if base_name in _EFFECTFUL_BUILTINS:
            self.effect_ops.add(dotted_name)
            return

        # Module prefix matching with proper boundaries
        for prefix in _EFFECTFUL_MODULE_PREFIXES:
            if self._matches_prefix(lowered, prefix):
                self.effect_ops.add(dotted_name)
                return

        # Tensor module matching (takes priority)
        for prefix in _TENSOR_MODULE_PREFIXES:
            if self._matches_prefix(lowered, prefix):
                self.tensor_ops.add(dotted_name)
                return

        # Orchestration module matching
        for prefix in _ORCHESTRATION_MODULE_PREFIXES:
            if self._matches_prefix(lowered, prefix):
                self.orchestration_ops.add(dotted_name)
                return

        # Fall back to whole-token keyword matching
        tokens = self._tokenize(dotted_name)

        for token in tokens:
            if token in _TENSOR_KEYWORDS:
                self.tensor_ops.add(dotted_name)
                return  # Don't also check orchestration

        for token in tokens:
            if token in _ORCHESTRATION_KEYWORDS:
                self.orchestration_ops.add(dotted_name)
                return

    def _matches_prefix(self, name: str, prefix: str) -> bool:
        """Check if name matches prefix with proper module boundary."""
        # Remove trailing dot from prefix for comparison
        prefix_base = prefix.rstrip(".")

        # Exact match
        if name == prefix_base:
            return True

        # Prefix with dot boundary
        if name.startswith(prefix_base + "."):
            return True

        return False

    def _tokenize(self, name: str) -> List[str]:
        """Split name into tokens for keyword matching."""
        return [t.lower() for t in _TOKEN_BOUNDARY.split(name) if t]

    def _extract_name(self, node: ast.AST) -> str:
        """Extract dotted name from AST node."""
        parts: List[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return ".".join(parts)


# -----------------------------------------------------------------------------
# Binding analysis
# -----------------------------------------------------------------------------


def _analyze_bindings(func: Callable[..., Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Analyze closure and global bindings for module references.

    This catches aliasing patterns like:
        import jax.numpy as np
        client = openai.OpenAI()

    Returns:
        Tuple of (tensor_evidence, orchestration_evidence, effect_evidence)
    """
    tensor_evidence: Set[str] = set()
    orchestration_evidence: Set[str] = set()
    effect_evidence: Set[str] = set()

    try:
        closure_vars = inspect.getclosurevars(func)
    except TypeError:
        return tensor_evidence, orchestration_evidence, effect_evidence

    # Check all referenced names
    all_vars = {
        **closure_vars.globals,
        **closure_vars.nonlocals,
        **{k: None for k in closure_vars.unbound},
    }

    for name, value in all_vars.items():
        if value is None:
            continue

        # Get module of the bound value
        module = getattr(value, "__module__", "")
        if not module:
            module = getattr(type(value), "__module__", "")

        if not module:
            continue

        module_lower = module.lower()

        # Check tensor modules
        for prefix in _JAX_TRACEABLE_PREFIXES:
            if _matches_module_prefix(module_lower, prefix):
                tensor_evidence.add(f"binding:{name}={module}")
                break

        for prefix in _NUMERIC_NON_JAX_PREFIXES:
            if _matches_module_prefix(module_lower, prefix):
                tensor_evidence.add(f"binding:{name}={module}")
                break

        # Check orchestration modules
        for prefix in _ORCHESTRATION_MODULE_PREFIXES:
            if _matches_module_prefix(module_lower, prefix):
                orchestration_evidence.add(f"binding:{name}={module}")
                break

        # Check effectful modules
        for prefix in _EFFECTFUL_MODULE_PREFIXES:
            if _matches_module_prefix(module_lower, prefix):
                effect_evidence.add(f"binding:{name}={module}")
                break

    return tensor_evidence, orchestration_evidence, effect_evidence


def _matches_module_prefix(module: str, prefix: str) -> bool:
    """Check if module matches prefix with proper boundary."""
    prefix_base = prefix.rstrip(".")
    return module == prefix_base or module.startswith(prefix_base + ".")


# -----------------------------------------------------------------------------
# Wrapper unwrapping
# -----------------------------------------------------------------------------


def _unwrap_chain(func: Callable[..., Any]) -> List[Callable[..., Any]]:
    """Return the unwrap chain from outer to inner callable.

    This handles decorators and functools.partial.
    """
    chain: List[Callable[..., Any]] = [func]
    current = func

    # Handle functools.partial
    while isinstance(current, functools.partial):
        chain.append(current.func)
        current = current.func

    # Handle decorated functions via __wrapped__
    seen: Set[int] = {id(current)}
    while hasattr(current, "__wrapped__"):
        wrapped = current.__wrapped__
        if id(wrapped) in seen:
            break
        seen.add(id(wrapped))
        chain.append(wrapped)
        current = wrapped

    return chain


def _get_markers_from_chain(
    chain: List[Callable[..., Any]]
) -> Tuple[Optional[bool], Optional[bool]]:
    """Check for markers on any function in the unwrap chain.

    Outermost markers win (first in chain).
    """
    is_tensor: Optional[bool] = None
    is_orchestration: Optional[bool] = None

    for func in chain:
        if is_tensor is None:
            is_tensor = getattr(func, "_xcs_is_tensor", None)
        if is_orchestration is None:
            is_orchestration = getattr(func, "_xcs_is_orchestration", None)
        if is_tensor is not None and is_orchestration is not None:
            break

    return is_tensor, is_orchestration


# -----------------------------------------------------------------------------
# Main analysis function
# -----------------------------------------------------------------------------


def analyze_operations_v2(func: Callable[..., Any]) -> OpDecision:
    """Analyze a function and return a rich classification decision.

    This is the new analysis API that returns the full OpDecision model.

    Priority order:
    1. Explicit markers (@mark_tensor, @mark_orchestration, @mark_hybrid)
    2. Closure/global binding analysis
    3. Module-based detection (function's __module__)
    4. AST-based heuristics with local import resolution

    Args:
        func: The callable to analyze.

    Returns:
        OpDecision with kind, traceability, effect risk, confidence, and evidence.
    """
    # Check cache first - skip for unhashable objects
    code_hash = _get_code_hash(func)
    try:
        with _CACHE_LOCK:
            cached = _ANALYSIS_CACHE.get(func)
            if cached is not None:
                cached_decision, cached_hash = cached
                if cached_hash == code_hash:
                    return cached_decision
    except TypeError:
        # Object is not hashable (e.g., equinox modules with JAX arrays)
        # Skip caching for this object
        pass

    # Unwrap function chain
    chain = _unwrap_chain(func)
    innermost = chain[-1] if chain else func

    # Priority 1: Check explicit markers
    is_marked_tensor, is_marked_orchestration = _get_markers_from_chain(chain)

    if is_marked_tensor is not None or is_marked_orchestration is not None:
        kind = _determine_kind_from_markers(is_marked_tensor, is_marked_orchestration)
        traceability = Traceability.JAX if is_marked_tensor else Traceability.UNKNOWN
        decision = OpDecision(
            kind=kind,
            jax_traceable=traceability,
            effect_risk=EffectRisk.LOW,
            confidence=1.0,
            tensor_evidence=frozenset({"marked"}) if is_marked_tensor else frozenset(),
            orchestration_evidence=(
                frozenset({"marked"}) if is_marked_orchestration else frozenset()
            ),
            reason="explicit_marker",
        )
        _cache_result(func, decision, code_hash)
        return decision

    # Priority 2: Binding analysis (catches aliasing)
    tensor_from_bindings, orch_from_bindings, effects_from_bindings = _analyze_bindings(
        innermost
    )

    # Priority 3: Module-based detection
    module = getattr(innermost, "__module__", "") or ""
    module_lower = module.lower()

    for prefix in _JAX_TRACEABLE_PREFIXES:
        if _matches_module_prefix(module_lower, prefix):
            decision = OpDecision(
                kind=OpKind.TENSOR,
                jax_traceable=Traceability.JAX,
                effect_risk=EffectRisk.LOW,
                confidence=0.9,
                tensor_evidence=frozenset({f"module:{module}"}),
                reason="module_jax_traceable",
            )
            _cache_result(func, decision, code_hash)
            return decision

    for prefix in _NUMERIC_NON_JAX_PREFIXES:
        if _matches_module_prefix(module_lower, prefix):
            decision = OpDecision(
                kind=OpKind.TENSOR,
                jax_traceable=Traceability.NON_JAX,
                effect_risk=EffectRisk.LOW,
                confidence=0.9,
                tensor_evidence=frozenset({f"module:{module}"}),
                reason="module_numeric_non_jax",
            )
            _cache_result(func, decision, code_hash)
            return decision

    for prefix in _ORCHESTRATION_MODULE_PREFIXES:
        if _matches_module_prefix(module_lower, prefix):
            decision = OpDecision(
                kind=OpKind.ORCHESTRATION,
                jax_traceable=Traceability.UNKNOWN,
                effect_risk=EffectRisk.HIGH,
                confidence=0.9,
                orchestration_evidence=frozenset({f"module:{module}"}),
                reason="module_orchestration",
            )
            _cache_result(func, decision, code_hash)
            return decision

    # Priority 4: AST-based heuristics
    try:
        source = inspect.getsource(innermost)
    except (OSError, TypeError):
        # Can't get source - return UNKNOWN (not hybrid!)
        decision = OpDecision(
            kind=OpKind.UNKNOWN,
            jax_traceable=Traceability.UNKNOWN,
            effect_risk=EffectRisk.MEDIUM,
            confidence=0.0,
            tensor_evidence=frozenset(tensor_from_bindings),
            orchestration_evidence=frozenset(orch_from_bindings),
            effect_evidence=frozenset(effects_from_bindings),
            reason="no_source",
        )
        _cache_result(func, decision, code_hash)
        return decision

    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        decision = OpDecision(
            kind=OpKind.UNKNOWN,
            jax_traceable=Traceability.UNKNOWN,
            effect_risk=EffectRisk.MEDIUM,
            confidence=0.0,
            reason="parse_error",
        )
        _cache_result(func, decision, code_hash)
        return decision

    visitor = _OperationVisitor()
    visitor.visit(tree)

    # Combine evidence
    all_tensor = tensor_from_bindings | visitor.tensor_ops
    all_orch = orch_from_bindings | visitor.orchestration_ops
    all_effects = effects_from_bindings | visitor.effect_ops

    # Determine kind
    has_tensor = bool(all_tensor)
    has_orch = bool(all_orch)

    if has_tensor and has_orch:
        kind = OpKind.HYBRID
    elif has_tensor:
        kind = OpKind.TENSOR
    elif has_orch:
        kind = OpKind.ORCHESTRATION
    else:
        kind = OpKind.NEUTRAL

    # Determine traceability
    traceability = _determine_traceability(all_tensor)

    # Determine effect risk
    effect_risk = _determine_effect_risk(all_effects, all_orch, visitor.is_async)

    # Calculate confidence
    evidence_count = len(all_tensor) + len(all_orch)
    confidence = min(0.9, 0.3 + 0.1 * evidence_count) if evidence_count > 0 else 0.1

    decision = OpDecision(
        kind=kind,
        jax_traceable=traceability,
        effect_risk=effect_risk,
        confidence=confidence,
        tensor_evidence=frozenset(all_tensor),
        orchestration_evidence=frozenset(all_orch),
        effect_evidence=frozenset(all_effects),
        reason="ast_analysis",
    )

    _cache_result(func, decision, code_hash)
    return decision


def _determine_kind_from_markers(
    is_tensor: Optional[bool], is_orchestration: Optional[bool]
) -> OpKind:
    """Determine OpKind from explicit markers."""
    if is_tensor and is_orchestration:
        return OpKind.HYBRID
    if is_tensor:
        return OpKind.TENSOR
    if is_orchestration:
        return OpKind.ORCHESTRATION
    return OpKind.NEUTRAL


def _determine_traceability(tensor_evidence: Set[str]) -> Traceability:
    """Determine JAX traceability from tensor evidence.

    Uses proper boundary matching to avoid false positives like
    'jnp' matching 'np.' as a substring.
    """
    has_jax = False
    has_non_jax = False

    for evidence in tensor_evidence:
        lowered = evidence.lower()

        # Check JAX-traceable prefixes with proper boundary matching
        for prefix in _JAX_TRACEABLE_PREFIXES:
            prefix_base = prefix.rstrip(".")
            if lowered == prefix_base or lowered.startswith(prefix_base + "."):
                has_jax = True
                break
            # Also check if evidence contains the prefix as a component
            # e.g., "jnp.sum" should match "jax" via alias resolution
            if prefix_base in ("jax", "jaxlib", "flax", "equinox", "optax"):
                # Check for common aliases
                if lowered.startswith("jnp.") or lowered.startswith("jax."):
                    has_jax = True
                    break

        # Check non-JAX prefixes with proper boundary matching
        for prefix in _NUMERIC_NON_JAX_PREFIXES:
            prefix_base = prefix.rstrip(".")
            # Must match at word boundary to avoid 'jnp' matching 'np'
            if lowered == prefix_base or lowered.startswith(prefix_base + "."):
                has_non_jax = True
                break

    if has_jax and not has_non_jax:
        return Traceability.JAX
    if has_non_jax and not has_jax:
        return Traceability.NON_JAX
    if has_jax and has_non_jax:
        # Mixed - prefer NON_JAX for safety
        return Traceability.NON_JAX
    return Traceability.UNKNOWN


def _determine_effect_risk(
    effect_evidence: Set[str], orch_evidence: Set[str], is_async: bool
) -> EffectRisk:
    """Determine effect risk level."""
    if is_async or effect_evidence:
        return EffectRisk.HIGH
    if orch_evidence:
        return EffectRisk.HIGH
    return EffectRisk.LOW


def _cache_result(func: Callable[..., Any], decision: OpDecision, code_hash: int) -> None:
    """Cache analysis result with code hash for invalidation."""
    try:
        with _CACHE_LOCK:
            _ANALYSIS_CACHE[func] = (decision, code_hash)
    except TypeError:
        # Object is not hashable (e.g., equinox modules with JAX arrays)
        # Skip caching for this object
        pass


# -----------------------------------------------------------------------------
# Legacy API compatibility
# -----------------------------------------------------------------------------


def analyze_operations(func: Callable[..., Any]) -> OperationSummary:
    """Analyze `func` and return detected operation categories.

    This is the legacy API that returns OperationSummary for backward
    compatibility. Internally uses analyze_operations_v2.

    Priority order (highest to lowest):
    1. Explicit markers (_xcs_is_tensor, _xcs_is_orchestration)
    2. Closure/global binding analysis
    3. Module-based detection (function's __module__)
    4. AST-based keyword heuristics

    Returns:
        OperationSummary with detected tensor and orchestration operations.
    """
    decision = analyze_operations_v2(func)

    # Convert to legacy format
    tensor_ops: Set[str] = set(decision.tensor_evidence) if decision.tensor_evidence else set()
    orchestration_ops: Set[str] = (
        set(decision.orchestration_evidence) if decision.orchestration_evidence else set()
    )

    return OperationSummary(
        tensor_ops=tensor_ops,
        orchestration_ops=orchestration_ops,
        _decision=decision,
    )


# -----------------------------------------------------------------------------
# Runtime JAX array detection (improved)
# -----------------------------------------------------------------------------


def is_jax_compatible_leaf(value: Any) -> bool:
    """Check if value is a JAX-compatible array leaf.

    This is stricter than is_jax_array and explicitly rejects torch/tf tensors.

    Returns:
        True only for JAX arrays, JAX tracers, and numpy arrays.
    """
    if value is None:
        return False

    # Check module first to reject torch/tf
    module_name = getattr(type(value), "__module__", "")
    if module_name.startswith(("torch", "tensorflow")):
        return False

    # JAX Tracer
    if jax_core is not None and isinstance(value, jax_core.Tracer):
        return True

    # JAX Array (various types)
    if jax is not None:
        jax_array_type = getattr(jax, "Array", None)
        if jax_array_type is not None and isinstance(value, jax_array_type):
            return True

    # ShapeDtypeStruct
    if _ShapeDtypeStruct and isinstance(value, _ShapeDtypeStruct):  # type: ignore[arg-type]
        return True

    # NumPy array
    if _NUMPY_AVAILABLE and isinstance(value, np.ndarray):
        return True

    # Abstract value with JAX-style aval
    aval = getattr(value, "aval", None)
    if aval is not None and hasattr(aval, "shape") and hasattr(aval, "dtype"):
        return True

    # JAX module check for other array types
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if "jax" in module_name:
            return True

    return False


def is_jax_array(value: Any) -> bool:
    """Return `True` when `value` looks like a JAX array.

    This is the legacy API. For stricter checking, use is_jax_compatible_leaf.
    """
    if value is None:
        return False

    # NumPy/JAX arrays expose the array protocol; keep this fast path intact.
    if hasattr(value, "__array__") and hasattr(value, "shape"):
        # Reject torch/tf tensors
        module_name = getattr(type(value), "__module__", "")
        if module_name.startswith(("torch", "tensorflow")):
            return False
        return True

    # Abstract tracing values (jit/vmap/eval_shape) do not implement __array__
    if _ShapeDtypeStruct and isinstance(value, _ShapeDtypeStruct):  # type: ignore[arg-type]
        return True

    if jax_core is not None and isinstance(value, jax_core.Tracer):
        return True

    aval = getattr(value, "aval", None)
    if aval is not None and hasattr(aval, "shape") and hasattr(aval, "dtype"):
        return True

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        module_name = getattr(type(value), "__module__", "")
        if "jax" in module_name:
            return True

    return False


# Limits for bounded tree traversal
_MAX_TREE_DEPTH = 10
_MAX_TREE_LEAVES = 10000


def has_jax_arrays(
    args: Iterable[Any], kwargs: Optional[Mapping[str, Any]] = None
) -> bool:
    """Return `True` when any argument contains a JAX array.

    Uses bounded traversal to prevent infinite loops and excessive scanning.
    """
    if jax_tree_util is not None:
        try:
            leaves = jax_tree_util.tree_leaves((tuple(args), dict(kwargs or {})))
        except Exception:
            return _has_jax_arrays_manual(args, kwargs)
        if any(is_jax_array(leaf) for leaf in leaves[:_MAX_TREE_LEAVES]):
            return True
        return _has_jax_arrays_manual(args, kwargs)
    return _has_jax_arrays_manual(args, kwargs)


def _has_jax_arrays_manual(
    args: Iterable[Any], kwargs: Optional[Mapping[str, Any]]
) -> bool:
    """Manual bounded traversal for JAX array detection."""
    visited: Set[int] = set()
    leaves_seen = [0]  # Mutable counter

    for item in args:
        if _contains_jax_array_bounded(item, visited, 0, leaves_seen):
            return True
    if kwargs:
        for value in kwargs.values():
            if _contains_jax_array_bounded(value, visited, 0, leaves_seen):
                return True
    return False


def _contains_jax_array_bounded(
    value: Any, visited: Set[int], depth: int, leaves_seen: List[int]
) -> bool:
    """Bounded check for JAX arrays in a value."""
    # Enforce limits
    if depth > _MAX_TREE_DEPTH:
        return False
    if leaves_seen[0] > _MAX_TREE_LEAVES:
        return False

    leaves_seen[0] += 1

    if is_jax_array(value):
        return True
    if isinstance(value, (str, bytes, type(None))):
        return False

    obj_id = id(value)
    if obj_id in visited:
        return False  # Cycle detected

    if isinstance(value, Mapping):
        visited.add(obj_id)
        return any(
            _contains_jax_array_bounded(v, visited, depth + 1, leaves_seen)
            for v in value.values()
        )

    if isinstance(value, (list, tuple, set)):
        visited.add(obj_id)
        return any(
            _contains_jax_array_bounded(item, visited, depth + 1, leaves_seen)
            for item in value
        )

    if is_dataclass(value):
        visited.add(obj_id)
        return any(
            _contains_jax_array_bounded(
                getattr(value, fld.name), visited, depth + 1, leaves_seen
            )
            for fld in fields(value)
        )

    if hasattr(value, "__dict__") and not isinstance(value, type):
        visited.add(obj_id)
        return any(
            _contains_jax_array_bounded(attr_value, visited, depth + 1, leaves_seen)
            for attr_value in value.__dict__.values()
        )

    return False


# -----------------------------------------------------------------------------
# Diagnostic / explain API
# -----------------------------------------------------------------------------


def explain(func: Callable[..., Any]) -> str:
    """Generate a diagnostic explanation of how a function is classified.

    This is useful for debugging classification issues and understanding
    why a function takes a particular execution path.

    Args:
        func: The callable to explain.

    Returns:
        Human-readable explanation string.
    """
    decision = analyze_operations_v2(func)

    lines = [
        f"Function: {getattr(func, '__name__', repr(func))}",
        f"Module: {getattr(func, '__module__', 'unknown')}",
        "",
        f"Classification: {decision.kind.name}",
        f"JAX Traceable: {decision.jax_traceable.name}",
        f"Effect Risk: {decision.effect_risk.name}",
        f"Confidence: {decision.confidence:.2f}",
        f"Reason: {decision.reason}",
        "",
    ]

    if decision.tensor_evidence:
        lines.append("Tensor Evidence:")
        for ev in sorted(decision.tensor_evidence)[:10]:
            lines.append(f"  - {ev}")

    if decision.orchestration_evidence:
        lines.append("Orchestration Evidence:")
        for ev in sorted(decision.orchestration_evidence)[:10]:
            lines.append(f"  - {ev}")

    if decision.effect_evidence:
        lines.append("Effect Evidence:")
        for ev in sorted(decision.effect_evidence)[:10]:
            lines.append(f"  - {ev}")

    lines.append("")
    lines.append("Recommendation:")

    if decision.kind == OpKind.UNKNOWN:
        lines.append("  Add @mark_tensor or @mark_orchestration for explicit classification")
    elif decision.kind == OpKind.NEUTRAL:
        if decision.can_jax_trace:
            lines.append("  Pure Python, will use JAX path if inputs are JAX arrays")
        else:
            lines.append("  Pure Python, add @mark_tensor if this should use JAX transforms")
    elif decision.kind == OpKind.TENSOR:
        if decision.jax_traceable == Traceability.JAX:
            lines.append("  Will use native JAX transforms")
        elif decision.jax_traceable == Traceability.NON_JAX:
            lines.append("  Uses non-JAX numeric library; JAX transforms will not work")
            lines.append("  Consider converting to jax.numpy for JAX compatibility")
    elif decision.kind == OpKind.ORCHESTRATION:
        lines.append("  Will use orchestration execution path")
        lines.append("  JAX transforms will error or use fallback execution")
    elif decision.kind == OpKind.HYBRID:
        lines.append("  Mixed tensor and orchestration operations")
        lines.append("  Consider splitting into separate functions for clarity")
        lines.append("  grad() will be restricted; use @mark_tensor on pure tensor parts")

    return "\n".join(lines)


__all__ = [
    # New API
    "OpKind",
    "Traceability",
    "EffectRisk",
    "OpDecision",
    "analyze_operations_v2",
    "is_jax_compatible_leaf",
    "explain",
    # Legacy API
    "OperationSummary",
    "analyze_operations",
    "is_jax_array",
    "has_jax_arrays",
]
