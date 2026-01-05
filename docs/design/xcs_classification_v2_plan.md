# XCS Operation Classification System — Robustness / Streamlining / Smarts Recommendations

**Goal:** Reduce wrong-path execution (crashes / unintended side effects), eliminate major performance cliffs, and improve inference quality without sacrificing the "explicit > implicit" philosophy.

## Executive summary

The current classifier conflates multiple concepts:

* **"Numerical / tensor-ish"** vs **"JAX-traceable / XLA-compilable"**
* **"Hybrid"** vs **"Unknown"**
* **"No indicators"** vs "Safe to run under JAX transforms"

These conflations create (1) correctness hazards (e.g., hybrid `grad` can execute network calls during tracing), (2) misdispatch (e.g., torch/tf treated as JAX-compile-safe), and (3) performance cliffs (pure-Python arithmetic functions routed to threadpool instead of JAX).

The fix is to keep the user-facing categories, but internally return a richer decision object: **kind + traceability + effect risk + confidence + evidence**.

---

## P0 — Correctness & safety (must-do)

### 1) Introduce `UNKNOWN` explicitly (stop treating "can't inspect" as "hybrid")

**Problem**

* In "no source / parse fail", you currently do:

  ```python
  OperationSummary(tensor_ops={"unknown"}, orchestration_ops={"unknown"})
  ```

  which makes the function appear *hybrid* (both sets non-empty). This is semantically wrong and can trigger hybrid policies (sanitization, staged assumptions, etc.) for truly unknown code.

**Recommendation**

* Add an explicit `kind` enum and avoid sentinel strings in evidence sets.

**Implementation sketch**

```python
class OpKind(Enum):
    TENSOR = auto()
    ORCH = auto()
    HYBRID = auto()
    NEUTRAL = auto()   # no evidence either way
    UNKNOWN = auto()   # inspection failed

@dataclass(frozen=True, slots=True)
class OperationSummary:
    kind: OpKind
    tensor_evidence: FrozenSet[str] = frozenset()
    orch_evidence: FrozenSet[str] = frozenset()
    confidence: float = 0.0
    reason: str = ""
```

**Acceptance tests**

* Builtin/C-extension function → `kind == UNKNOWN`, not HYBRID.
* "Unknown" never appears in evidence sets.

---

### 2) Split "tensor-only" into "JAX-traceable" vs "non-JAX numeric" (prevents JAX crashes)

**Problem**

* You classify `numpy.`, `torch.`, `tensorflow.`, `scipy.` as "tensor-only" and then treat "tensor-only" as eligible for JAX transforms and "Direct JAX/XLA compilation".
* This is not generally valid:

  * `torch.*` and `tensorflow.*` are not JAX-traceable.
  * `scipy.*` is not `jax.scipy.*`.
  * `numpy.*` ops inside the function are **not** traceable under `jax.jit` unless they're actually `jax.numpy` calls.

**Recommendation**

* Keep the existing 3 categories for the UX, but internally add a second axis:

  * `jax_traceable: bool` (or `Traceability.JAX / Traceability.NON_JAX / Traceability.UNKNOWN`)
* Only attempt `jax.jit/vmap/grad` when `jax_traceable` is True.

**Concrete changes**

* Replace `_TENSOR_MODULE_PREFIXES` with two lists:

  * `_JAX_TRACEABLE_PREFIXES = ("jax.", "jaxlib.", "jax.numpy.", "jax.scipy.", "flax.", "equinox.", "optax.", ...)` (whatever you consider safe)
  * `_NUMERIC_NON_JAX_PREFIXES = ("numpy.", "torch.", "tensorflow.", "scipy.", ...)`
* "tensor-only" can still be true for torch, but that must not imply "try JAX".

**Acceptance tests**

* `torch.nn.functional.relu` → `kind == TENSOR`, `jax_traceable == False`.
* Function calling `numpy.dot` → `jax_traceable == False`.
* Function calling `jax.numpy.dot` → `jax_traceable == True`.

---

### 3) Fix runtime JAX input detection (`is_jax_array` is too permissive)

**Problem**

* Current check:

  ```python
  if hasattr(value, "__array__") and hasattr(value, "shape"): return True
  ```

  will accept many non-JAX tensors (torch/tf) and other array-likes, causing attempted JAX transforms on incompatible inputs or expensive implicit conversions.

**Recommendation**

* Tighten to *actual* JAX or NumPy primitives you explicitly support, and explicitly reject torch/tf.

**Implementation sketch**

```python
import numpy as np
import jax
from jax.core import Tracer

def is_jax_compatible_leaf(x) -> bool:
    if isinstance(x, Tracer):
        return True
    if isinstance(x, getattr(jax, "Array", ())):
        return True
    if isinstance(x, jax.ShapeDtypeStruct):
        return True
    if isinstance(x, np.ndarray):
        return True
    mod = type(x).__module__
    if mod.startswith(("torch", "tensorflow")):
        return False
    return False
```

**Acceptance tests**

* torch tensor input does *not* pass `_inputs_can_use_native_jax`.
* numpy ndarray passes.
* jax.Array and Tracer pass.

---

### 4) Hybrid `grad` must not execute orchestration calls during tracing

**Problem**

* Current hybrid `grad` wrapper:

  ```python
  def safe_func(...):
      result = func(...)     # can execute LLM/network during JAX tracing
      return _sanitize(...)
  return jax.grad(safe_func)(...)
  ```

  This can:

  * Execute network/API calls during tracing (possibly multiple times),
  * Break determinism / caching assumptions,
  * Be catastrophic in production.

**Recommendation (choose one; I'd do A now, B later)**

* **A (safe default):** disallow `grad` on hybrid unless user explicitly marks it safe or provides a staging plan.

  * E.g., require `@mark_hybrid(grad_policy="stop_orch")` or `@mark_orchestration_nondiff`.
* **B (correct long-term):** stage hybrid functions into tensor subgraph + orchestration subgraph (using your IR builder), execute orchestration outside the JAX-traced region, and feed results as static inputs (or via `stop_gradient`) into tensor region.

**Acceptance tests**

* Hybrid function with an `openai` call: `grad(f)` does not make any network call during trace.
* In strict mode, `grad(hybrid)` raises with actionable message unless explicitly allowed.

---

### 5) Add "effect risk" detection (I/O, network, subprocess) and use it as a hard gate for any speculative JAX tracing

**Problem**

* If you ever add speculative "try JAX fast path" for NEUTRAL/UNKNOWN functions (recommended for performance), you must avoid tracing effectful code.

**Recommendation**

* Add a cheap, conservative "effectful" signal:

  * Builtins: `open`, `print`, `input`, `exec`, `eval`, `compile`, `__import__`
  * Modules: `os.`, `pathlib.`, `subprocess.`, `socket.`, `requests.`, `httpx.`, `aiohttp.`, DB libs, cloud SDKs
  * Async (`async def`, `await`) is a strong effect hint

Return `effect_risk: LOW/MED/HIGH` and treat `HIGH` as "never JAX-trace".

**Acceptance tests**

* Function containing `open()` or `requests.get()` → `effect_risk == HIGH`, `jax_traceable` path never used.

---

## P1 — Robustness & inference quality (big wins, low risk)

### 6) Add closure/globals binding detection (beats keyword matching, works without source)

**Problem**

* AST keyword heuristics miss common aliasing patterns:

  ```python
  import jax.numpy as np
  def f(x): return np.sum(x)
  ```

  and don't help when source is unavailable.

**Recommendation**

* Inspect function bytecode names and resolve global/freevar bindings (`inspect.getclosurevars`) to infer referenced modules/objects.

**Acceptance tests**

* Alias `np` bound to `jax.numpy` is correctly recognized as JAX-traceable.
* `client` bound to `openai.OpenAI()` instance is recognized as orchestration.

---

### 7) Support local imports inside functions (AST import alias map)

**Problem**

* Binding detection won't see **local** imports inside the function body, and AST string heuristics can't resolve them.

**Recommendation**

* In AST visitor, record `Import` / `ImportFrom` inside the function and build an alias map:

  * `import jax.numpy as jnp` → `jnp -> jax.numpy`
  * `from openai import OpenAI as Client` → `Client -> openai.OpenAI`

Use this alias map when extracting dotted names (`jnp.sum` resolves to `jax.numpy.sum`).

**Acceptance tests**

* Local import patterns classify correctly without executing the function.

---

### 8) Fix prefix and keyword matching to avoid false positives (boundary-aware)

**Problems**

* `lowered.startswith(prefix.lower().rstrip("."))` is buggy:

  * prefix `"jax."` becomes `"jax"` and matches `"jaxx.foo"`.
* `token in lowered` substring matching is noisy (`array`, `grad`, etc.).

**Recommendation**

* Prefix: match module boundary (`name == prefix` or `name.startswith(prefix + ".")`)
* Keywords: tokenize on `.`/`_`/non-alnum and match whole tokens
* Consider demoting generic tokens (`array`, `tensor`) to *weak* signals or removing them entirely.

**Acceptance tests**

* `jaxx.*` does not classify as JAX.
* `upgrade()` does not trigger `grad` token.

---

### 9) Handle wrappers/partials/callables correctly (avoid analyzing the wrong thing)

**Problem**

* Decorators and `functools.partial` can cause you to analyze wrapper code instead of user code.
* If you call `inspect.unwrap`, you might skip marker attributes that live on outer wrappers.

**Recommendation**

* Implement `unwrap_chain(func)` returning the chain `[outer, ..., inner]`.
* Marker detection should check the entire chain (outermost wins).
* Analysis should operate on the most "real" callable (usually inner), but carry forward explicit markers from any wrapper.

**Acceptance tests**

* Markers on outer wrappers are honored.
* Partials are analyzed using `partial.func`.

---

## P2 — Performance & scalability

### 10) Cache analysis results with weak references + code hashing

**Problem**

* `inspect.getsource` + `ast.parse` is expensive and shows up repeatedly in notebooks / dynamic wrapping / hot reloads.
* Caching by function identity alone is brittle if code changes.

**Recommendation**

* Use a `WeakKeyDictionary` keyed by the *outer callable object*, storing:

  * `summary`
  * a lightweight "version stamp" (e.g., `id(func.__code__)` or `(co_code hash, co_consts hash)`)

Also add a small lock for thread-safety.

**Acceptance tests**

* Same function analyzed multiple times hits cache.
* Redefining function invalidates cache.

---

### 11) Make runtime tree traversal cycle-safe + bounded

**Problem**

* `_has_jax_arrays_manual` can infinite-loop on cyclic structures.
* Scanning arbitrary `__dict__` can be huge.

**Recommendation**

* Track `visited_ids` and cap:

  * max depth (e.g., 10)
  * max leaves visited (e.g., 10k)
* Fail safe (return False / Unknown) if limits exceeded.

**Acceptance tests**

* Cyclic structures don't hang.
* Large objects don't blow latency.

---

## P3 — Developer experience & observability (pays for itself)

### 12) Add `xcs.explain(func)` + warn-once for fallback paths

**Problem**

* When performance is suboptimal or behavior surprises users, they need immediate evidence and a fix path.

**Recommendation**

* Provide a tool that prints:

  * kind, traceability, confidence, effect risk
  * evidence (top N)
  * recommendation ("add @mark_tensor" / "function appears effectful; won't jit")

Also add warn-once when:

* `UNKNOWN` routes to orchestration path,
* `TENSOR` but `jax_traceable == False` blocks JAX transforms,
* hybrid grad is disallowed.

---

### 13) Add "strict mode" and "lenient mode"

**Recommendation**

* Strict mode: require explicit markers for anything ambiguous in `jit/grad/vmap`.
* Lenient mode: allow speculative JAX fast path when `effect_risk` is LOW and runtime inputs are JAX-compatible (with safe fallback).

This keeps prod safe while enabling experimentation in notebooks.

---

## Suggested revised internal decision model

Keep user-facing categories, but return something like:

```python
@dataclass(frozen=True, slots=True)
class OpDecision:
    kind: OpKind                 # TENSOR/ORCH/HYBRID/NEUTRAL/UNKNOWN
    jax_traceable: Optional[bool]
    effect_risk: EffectRisk      # LOW/MED/HIGH
    confidence: float            # 0..1
    evidence: tuple[str, ...]
```

Then transforms decide:

* `vmap/jit/grad`:

  * if explicit marker says tensor → attempt JAX only if `jax_traceable != False` and `effect_risk != HIGH`
  * if orchestration → threadpool
  * if neutral/unknown → attempt JAX only in lenient mode and only when `effect_risk == LOW` and runtime inputs are JAX-compatible; otherwise threadpool
* `grad`:

  * hybrid requires explicit policy or staging; otherwise error

---

## Minimal "implementation plan" (sequenced)

1. **P0.1:** Add `OpKind.UNKNOWN` + remove `"unknown"` sentinel evidence.
2. **P0.2:** Add `jax_traceable` axis; fix module prefix tables (split JAX vs non-JAX numeric).
3. **P0.3:** Fix runtime `is_jax_array` → `is_jax_compatible_leaf`.
4. **P0.4:** Add effect-risk detection + hard gate.
5. **P0.5:** Hybrid grad policy: disallow by default or stage (short-term disallow with explicit override).
6. **P1:** Add bindings detection + local import alias map.
7. **P1:** Fix matching boundaries and unwrap chain handling.
8. **P2:** Add caching + bounded traversal.
9. **P3:** Add `explain()` + warn-once + strict/lenient mode flags.

---

## Key tests to add (high signal)

* Alias global + local import:

  * `import jax.numpy as np` and `import jax.numpy as jnp` inside function
* Operator-only pure python:

  * `def f(x): return x + 1` → uses JAX path when given JAX arrays
* Torch/tf inputs:

  * should not be treated as JAX-compatible runtime leaves
* "Unknown source" function:

  * builtins/C ext → `UNKNOWN`, no fake HYBRID
* Hybrid grad:

  * must not execute orchestration calls during trace; should error unless explicitly allowed
