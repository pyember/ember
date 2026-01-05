# XCS Operation Classification System

## Design Document

**Status**: Implemented
**Authors**: Ember Team
**Last Updated**: 2024-12

---

## 1. Overview

The XCS (eXecutable Computation Specification) system needs to classify functions into three categories to determine the appropriate execution strategy:

| Category | Description | Execution Strategy |
|----------|-------------|-------------------|
| **Tensor-only** | Pure numerical/array operations | Direct JAX/XLA compilation |
| **Orchestration-only** | LLM/API calls, I/O operations | Thread-pool parallelism |
| **Hybrid** | Mix of both | Staged execution with sanitization |

This classification directly impacts:
- Whether `jax.jit`, `jax.vmap`, `jax.grad` can be applied
- Whether parallel execution uses XLA fusion or thread pools
- How gradients flow through hybrid computations

---

## 2. Design Philosophy

### 2.1 Principle: Explicit Over Implicit

The system follows a priority order that prefers explicit user declarations over heuristic inference:

```
Priority 1: Explicit Markers     (@mark_tensor, @mark_orchestration, @mark_hybrid)
    ↓
Priority 2: Module Detection     (function's __module__ attribute)
    ↓
Priority 3: AST Keyword Heuristics   (source code analysis)
    ↓
Priority 4: Type Annotations     (signature inspection)
```

**Rationale**: Users know their code better than any heuristic. When users provide explicit annotations, we trust them completely. Heuristics are a fallback, not the primary mechanism.

### 2.2 Principle: Conservative on Ambiguity

When the system cannot confidently classify a function, it defaults to treating it as **hybrid** (both tensor and orchestration). This is the safest option because:

- Hybrid functions use orchestration batching (thread pools), which works for all code
- This prevents incorrect XLA compilation of non-tensor code (which would crash)
- It never produces incorrect results, only potentially suboptimal performance

### 2.3 Principle: Module Awareness Over Keywords

A keyword like "model" appears in both:
- `torch.nn.Module` (tensor operation)
- `openai.Model` (orchestration operation)

The system uses **module-aware matching**: if a dotted name starts with a known module prefix, that prefix determines classification, overriding any keyword matching.

---

## 3. Implementation Architecture

### 3.1 Core Data Structure

```python
@dataclass(slots=True)
class OperationSummary:
    """Summary of detected operation types inside a function."""

    tensor_ops: Set[str]        # Names/identifiers classified as tensor ops
    orchestration_ops: Set[str]  # Names/identifiers classified as orchestration

    @property
    def only_tensor_ops(self) -> bool:
        return self.has_tensor_ops and not self.has_orchestration_ops

    @property
    def only_orchestration_ops(self) -> bool:
        return self.has_orchestration_ops and not self.has_tensor_ops

    @property
    def is_hybrid(self) -> bool:
        return self.has_tensor_ops and self.has_orchestration_ops
```

### 3.2 Entry Point: `analyze_operations(func)`

The main classification function located in `src/ember/xcs/compiler/analysis.py`:

```python
def analyze_operations(func: Callable[..., Any]) -> OperationSummary:
    """Analyze func and return detected operation categories.

    Priority order (highest to lowest):
    1. Explicit markers (_xcs_is_tensor, _xcs_is_orchestration)
    2. Module-based detection (function's __module__)
    3. AST-based keyword heuristics
    """
```

---

## 4. Detection Mechanisms

### 4.1 Priority 1: Explicit Markers

Users can explicitly mark functions using decorators from `ember.api.decorators`:

```python
from ember.api.decorators import mark_tensor, mark_orchestration, mark_hybrid

@mark_tensor
def compute_loss(x: jnp.ndarray, y: jnp.ndarray) -> float:
    return jnp.mean((x - y) ** 2)

@mark_orchestration
def call_llm(prompt: str) -> str:
    return client.complete(prompt)

@mark_hybrid
def embed_and_query(text: str) -> jnp.ndarray:
    embedding = encode(text)  # tensor
    result = llm_call(text)    # orchestration
    return process(embedding, result)
```

**Implementation Details:**
- `@mark_tensor` sets `func._xcs_is_tensor = True`, `func._xcs_is_orchestration = False`
- `@mark_orchestration` sets `func._xcs_is_orchestration = True`, `func._xcs_is_tensor = False`
- `@mark_hybrid` sets both to `True`
- When any marker attribute is present, `analyze_operations` returns immediately without heuristic analysis

**Additional Marker: `@mark_pytree_safe`**

Controls whether function outputs are valid JAX pytrees (for vmap/pmap eligibility):

```python
@mark_pytree_safe(False)
def get_file_handle(path: str) -> IO:
    return open(path)  # File handles are not pytree-compatible
```

### 4.2 Priority 2: Module-Based Detection

If the function's `__module__` attribute starts with a known prefix, classification is determined by that prefix:

**Tensor Module Prefixes:**
```python
_TENSOR_MODULE_PREFIXES = (
    "jax.",
    "jaxlib.",
    "numpy.",
    "torch.",
    "tensorflow.",
    "scipy.",
)
```

**Orchestration Module Prefixes:**
```python
_ORCHESTRATION_MODULE_PREFIXES = (
    "anthropic.",
    "openai.",
    "langchain.",
    "litellm.",
    "ember.gateway.",
    "ember.operators.",
)
```

**Example:**
- A function with `__module__ = "torch.nn.functional"` → classified as **tensor-only**
- A function with `__module__ = "openai.resources"` → classified as **orchestration-only**

### 4.3 Priority 3: AST-Based Keyword Heuristics

For user-defined functions without explicit markers, the system parses the source code and looks for indicators.

**Implementation: `_OperationVisitor` (AST NodeVisitor)**

```python
class _OperationVisitor(ast.NodeVisitor):
    """AST visitor that records tensor and orchestration indicators."""

    def visit_Call(self, node: ast.Call) -> Any:
        name = self._extract_name(node.func)  # e.g., "jax.numpy.dot"
        self._record_name(name)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        name = self._extract_name(node)
        self._record_name(name)
```

**Module-Aware Matching in AST:**

The visitor first checks if a dotted name matches a known module prefix:

```python
def _record_name(self, dotted_name: str) -> None:
    lowered = dotted_name.lower()

    # Check tensor module prefixes first
    for prefix in _TENSOR_MODULE_PREFIXES:
        if lowered.startswith(prefix.lower().rstrip(".")):
            self.tensor_ops.add(dotted_name)
            return  # Don't also check keywords

    # Check orchestration module prefixes
    for prefix in _ORCHESTRATION_MODULE_PREFIXES:
        if lowered.startswith(prefix.lower().rstrip(".")):
            self.orchestration_ops.add(dotted_name)
            return  # Don't also check keywords

    # Fall back to keyword matching
    if any(token in lowered for token in _TENSOR_KEYWORDS):
        self.tensor_ops.add(dotted_name)
    if any(token in lowered for token in _ORCHESTRATION_KEYWORDS):
        self.orchestration_ops.add(dotted_name)
```

**Current Keyword Sets:**

```python
# Tensor keywords - unambiguous, only appear in numerical computing
_TENSOR_KEYWORDS = {
    "jax", "jnp", "numpy", "ndarray", "tensor", "array",
    "matmul", "einsum", "lax", "grad", "vmap", "pmap", "scan",
    "torch", "pytorch", "tensorflow", "tf",
}

# Orchestration keywords - unambiguous LLM/API indicators
_ORCHESTRATION_KEYWORDS = {
    "llm", "anthropic", "openai", "prompt", "completion", "chat",
    "mcp", "langchain", "litellm", "gemini", "cohere",
}
```

**Intentionally Excluded Keywords:**
- `model` - Ambiguous (torch.nn.Module vs OpenAI model)
- `call` - Too generic (function calls everywhere)
- `api` - Too generic (REST APIs, internal APIs)
- `router` - Internal Ember concept
- `tool` - Too generic

### 4.4 Priority 4: Type Annotation Inspection

As a final signal, the system inspects function type annotations:

```python
signature = inspect.signature(func)
hints = [repr(param.annotation) for param in signature.parameters.values()]

for hint in hints:
    lowered = hint.lower()
    if any(token in lowered for token in ("array", "tensor", "ndarray", "jax")):
        summary.tensor_ops.add(hint)
```

Only unambiguous type hints are used. The following were removed as too ambiguous:
- `str` - Orchestration might use strings, but so does everything else
- `model` - Same ambiguity as the keyword
- `operator` - Too generic

---

## 5. Runtime Type Detection

Beyond static analysis, the system can inspect runtime argument types:

### 5.1 `is_jax_array(value)`

Detects JAX-compatible arrays at runtime:

```python
def is_jax_array(value: Any) -> bool:
    # Fast path: NumPy/JAX array protocol
    if hasattr(value, "__array__") and hasattr(value, "shape"):
        return True

    # JAX abstract tracing values
    if isinstance(value, jax_core.Tracer):
        return True

    # ShapeDtypeStruct (used in jax.eval_shape)
    if isinstance(value, ShapeDtypeStruct):
        return True

    # Objects with aval attribute (JAX tracing)
    aval = getattr(value, "aval", None)
    if aval and hasattr(aval, "shape") and hasattr(aval, "dtype"):
        return True

    return False
```

### 5.2 `has_jax_arrays(args, kwargs)`

Recursively checks if any argument contains a JAX array:

```python
def has_jax_arrays(args, kwargs) -> bool:
    # Use JAX tree_leaves for efficient traversal
    leaves = jax_tree_util.tree_leaves((args, kwargs))
    if any(is_jax_array(leaf) for leaf in leaves):
        return True

    # Fall back to manual traversal for non-pytree containers
    return _has_jax_arrays_manual(args, kwargs)
```

The manual traversal handles:
- Mappings (dict, etc.)
- Sequences (list, tuple, set)
- Dataclasses
- Objects with `__dict__`

---

## 6. Dispatch Decisions

### 6.1 `vmap` Transform

```python
def vmap(func, *, in_axes=0, out_axes=0, ...):
    ops = analyze_operations(func)

    if ops.only_tensor_ops:
        jax_vmapped = jax.vmap(func, in_axes, out_axes, ...)

        @wraps(func)
        def tensor_wrapper(*args, **kwargs):
            # Runtime check before using JAX path
            if _inputs_can_use_native_jax(args, kwargs):
                return jax_vmapped(*args, **kwargs)
            # Fall back to orchestration batching
            return _execute_orchestration_batch(func, args, kwargs, ...)

        return tensor_wrapper

    # Orchestration or hybrid: use thread-pool batching
    return orchestration_wrapper
```

**Key Insight**: Even if static analysis says "tensor-only", the runtime check `_inputs_can_use_native_jax` validates that actual inputs are JAX-compatible. This handles cases where static analysis was correct but the function is called with unexpected argument types.

### 6.2 `grad` Transform

```python
def grad(func, *, argnums=0, has_aux=False, ...):
    ops = analyze_operations(func)

    if ops.only_orchestration_ops:
        # Cannot differentiate through LLM calls
        raise XCSError("Cannot compute gradients through orchestration-only functions.")

    if ops.only_tensor_ops:
        # Direct JAX grad
        return jax.grad(func, argnums, has_aux, ...)

    # Hybrid: wrap non-array outputs as static
    @wraps(func)
    def wrapper(*args, **kwargs):
        def safe_func(*inner_args, **inner_kwargs):
            result = func(*inner_args, **inner_kwargs)
            return _sanitize_non_array_leaves(result)  # Wrap non-JAX values

        return jax.grad(safe_func, argnums, ...)(*args, **kwargs)

    return wrapper
```

**Gradient Through Hybrid Functions**: Non-JAX values (like strings from LLM calls) are wrapped in `StaticWrapper`, which JAX treats as a static leaf. Gradients are zero through these paths, which is mathematically correct.

### 6.3 `jit` Decorator

```python
def jit(func, *, config=None):
    # Build IR graph via tracing
    graph, result = BUILDER.trace_with_result(func, *args, **kwargs)

    # Analyze for parallelism
    analysis = ANALYZER.analyze(graph)

    # Check if graph is pure tensor (can skip IR execution)
    is_pure_tensor = _graph_is_pure_tensor(graph)

    if is_pure_tensor:
        # Bypass IR execution, just call function directly
        pure_graph_detected = True
        return result

    # Execute through IR with potential parallelism
    return ENGINE.execute(graph, args, kwargs, parallelism=analysis, config=config)
```

---

## 7. Edge Cases and Error Handling

### 7.1 Source Not Available

Built-in functions, C extensions, and some dynamically generated functions don't have accessible source:

```python
try:
    source = inspect.getsource(func)
except (OSError, TypeError):
    # Conservative: return hybrid
    return OperationSummary(
        tensor_ops={"unknown"},
        orchestration_ops={"unknown"},
    )
```

### 7.2 Syntax Errors in Source

If the function source can't be parsed (e.g., partial code):

```python
try:
    tree = ast.parse(source)
except SyntaxError:
    return OperationSummary(
        tensor_ops={"unknown"},
        orchestration_ops={"unknown"},
    )
```

### 7.3 No Indicators Found

If AST analysis finds no tensor or orchestration keywords:

```python
# Returns empty sets - neither tensor nor orchestration
OperationSummary(tensor_ops=set(), orchestration_ops=set())

# Properties:
# - only_tensor_ops: False
# - only_orchestration_ops: False
# - is_hybrid: False
```

This represents pure Python code without any JAX or LLM calls, which can run through either path.

---

## 8. Caching and Performance

### 8.1 Analysis Caching

`analyze_operations` is called once per function per transform. Results are not explicitly cached at the analysis level, but:

- JIT caches the full `_CacheEntry` including `is_pure_tensor`
- Transforms are typically applied at module load time, not per-call

### 8.2 Axis Cache

For vmap/pmap axis normalization:

```python
_AXIS_CACHE: Dict[
    Tuple[Any, int, Tuple[str, ...]],
    Tuple[Tuple[Optional[int], ...], Tuple[Tuple[str, Optional[int]], ...]],
] = {}
```

---

## 9. Future Improvements

### 9.1 Considered but Not Implemented

| Improvement | Rationale for Deferral |
|-------------|----------------------|
| **Runtime validation warnings** | Explicit markers solve the problem more cleanly |
| **Machine learning classifier** | Heuristics are sufficient; ML adds complexity |
| **Per-call dynamic dispatch** | Performance overhead; caching is preferred |
| **User-extensible keyword sets** | Complexity; explicit markers are the escape hatch |

### 9.2 Potential Enhancements

1. **Import Graph Analysis**: Track imports to detect transitive dependencies on tensor/orchestration libraries.

2. **Call Graph Depth Limiting**: For deeply nested calls, limit analysis depth to prevent slowdowns.

3. **Confidence Scores**: Instead of binary classification, return confidence levels that transforms could use for optimization decisions.

4. **Static Type Inference**: Use type stubs and type checkers to infer types without runtime execution.

5. **Caching at Analysis Level**: Memoize `analyze_operations` results per function identity.

---

## 10. Testing Strategy

### 10.1 Unit Tests

Located in `tests/unit/xcs/test_analysis_classification.py`:

- **Explicit marker tests**: Verify markers override heuristics
- **Ambiguous keyword tests**: Verify removed keywords don't cause false positives
- **Module-aware matching tests**: Verify torch.* and openai.* are correctly classified
- **Edge case tests**: Built-in functions, empty functions, pure Python

### 10.2 Integration Tests

Located in `tests/unit/xcs/test_xcs_transformations.py`:

- **Transform dispatch tests**: Verify vmap/pmap/grad use correct execution path
- **Hybrid function tests**: Verify gradients flow correctly through mixed code
- **Error handling tests**: Verify orchestration-only grad raises appropriate error

---

## 11. Related Documentation

- `docs/design/xcs_hybrid_execution.md` - Execution engine design
- `src/ember/xcs/README.md` - XCS module overview
- `src/ember/api/decorators.py` - Decorator implementations

---

## Appendix A: Complete Keyword Reference

### A.1 Tensor Keywords (Current)

| Keyword | Rationale |
|---------|-----------|
| `jax` | JAX framework |
| `jnp` | JAX numpy alias |
| `numpy` | NumPy library |
| `ndarray` | NumPy array type |
| `tensor` | General tensor term |
| `array` | General array term |
| `matmul` | Matrix multiplication |
| `einsum` | Einstein summation |
| `lax` | JAX low-level API |
| `grad` | Gradient computation |
| `vmap` | Vectorization |
| `pmap` | Parallelization |
| `scan` | Sequential scan |
| `torch` | PyTorch framework |
| `pytorch` | PyTorch framework |
| `tensorflow` | TensorFlow framework |
| `tf` | TensorFlow alias |

### A.2 Orchestration Keywords (Current)

| Keyword | Rationale |
|---------|-----------|
| `llm` | Large language model |
| `anthropic` | Anthropic API |
| `openai` | OpenAI API |
| `prompt` | LLM prompt |
| `completion` | LLM completion |
| `chat` | Chat API |
| `mcp` | Model Context Protocol |
| `langchain` | LangChain framework |
| `litellm` | LiteLLM library |
| `gemini` | Google Gemini |
| `cohere` | Cohere API |

### A.3 Removed Keywords (Historical)

| Keyword | Reason for Removal |
|---------|-------------------|
| `model` | Ambiguous: torch.nn.Module vs OpenAI model |
| `call` | Too generic: appears in all code |
| `api` | Too generic: REST APIs, internal APIs |
| `router` | Internal Ember concept |
| `tool` | Too generic |
| `dot` | Ambiguous: string method, math operation |

---

## Appendix B: Decision Tree

```
analyze_operations(func)
│
├─ Has _xcs_is_tensor or _xcs_is_orchestration attribute?
│   └─ YES → Return based on marker values (DONE)
│
├─ func.__module__ starts with tensor prefix?
│   └─ YES → Return tensor-only (DONE)
│
├─ func.__module__ starts with orchestration prefix?
│   └─ YES → Return orchestration-only (DONE)
│
├─ Can get source code?
│   └─ NO → Return hybrid/unknown (DONE)
│
├─ Parse AST and visit nodes
│   ├─ For each Call/Attribute node:
│   │   ├─ Name starts with tensor module prefix?
│   │   │   └─ YES → Add to tensor_ops, skip keywords
│   │   ├─ Name starts with orchestration module prefix?
│   │   │   └─ YES → Add to orchestration_ops, skip keywords
│   │   ├─ Name contains tensor keyword?
│   │   │   └─ YES → Add to tensor_ops
│   │   └─ Name contains orchestration keyword?
│   │       └─ YES → Add to orchestration_ops
│   │
│   └─ Return OperationSummary(tensor_ops, orchestration_ops)
```
