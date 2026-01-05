# Ember XCS (Execution Control System)

XCS is Ember's execution layer for hybrid AI workloads. It traces Python
functions, builds an intermediate representation (IR), discovers parallelism,
and executes the resulting graph with minimal user configuration.

## Quick Start

```python
from ember.xcs import jit

@jit
def predict(batch):
    return model(batch)

outputs = predict(input_batch)
```

- First call traces `predict`, builds an IR graph, and runs sequentially.
- Subsequent calls reuse the graph and automatically parallelize independent
  work when possible.
- Set `Config(profile=True)` to record profiler data during execution.

## Package Layout

```
ember/xcs/
├── api/            # Public API layer (jit + transformations + stats)
├── compiler/       # IR data structures, tracing, and analysis
├── runtime/        # Execution engine and profiler
├── tracing/        # sys.settrace-based tracer
├── utils/          # Shared helpers (e.g., JAX pytree registration)
├── errors.py       # XCSError definition
├── config.py       # Config dataclass + presets
├── jit.py          # Compatibility re-export of api.jit
└── transformations.py  # Compatibility re-export of api.transforms
```

Legacy imports such as `ember.xcs._internal.analysis` continue to work; they now
forward to the new modules for backwards compatibility.

## Public API

The public surface is intentionally small:

```python
from ember.xcs import Config, Presets, XCSError
from ember.xcs import grad, jit, pmap, scan, vmap
from ember.xcs import get_jit_stats
```

### Configuration

```python
from ember.xcs import Config, jit

cfg = Config(max_workers=2, profile=True)

@jit(config=cfg)
def expensive(x):
    return heavy_pipeline(x)
```

`Config` values are validated eagerly. Use `Presets.SECURE`, `Presets.DEBUG`,
`Presets.LIGHTWEIGHT`, or `Presets.SERIAL` for common scenarios.

### Transformations

- `vmap` and `pmap` use JAX when a function is tensor-only, otherwise they batch
  orchestration calls (LLM/tools) using a thread pool that respects
  `Config.max_workers`.
- `scan` falls back to a Python loop when orchestration is present.
- `grad` forwards to `jax.grad` for tensor workloads and guards against pure
  orchestration, throwing an `XCSError` with guidance.

### Marking Orchestration Helpers

Heuristics still catch most orchestration helpers, but you can now opt-in
explicitly:

```python
from ember.api.decorators import mark_orchestration

@mark_orchestration
def call_model(prompt: str) -> str:
    return llm_client.complete(prompt)
```

The IR builder tags decorated callables with `is_orchestration`, giving the
analyzer and runtime unambiguous hints without relying on name matching.

### Profiling

Profiling is opt-in. Enable it per call or globally via an environment
variable:

```python
@jit(config=Config(profile=True))
def pipeline(x):
    ...

print(get_jit_stats(pipeline))
# {'calls': 3.0, 'total_time_ms': 41.2, ...}
```

Set `EMBER_XCS_PROFILE=1` to profile all jitted functions by default.
Use `ember.xcs.api.stats.reset_stats()` to clear recorded data.

## How It Works

1. `api.jit` uses a singleton `IRBuilder` (runtime tracing) and a
   `ParallelismAnalyzer` to produce an `IRGraph`.
2. The `ExecutionEngine` executes the graph sequentially or submits independent
   groups to a thread pool respecting `Config.max_workers`.
3. The `Profiler` records optional execution metrics (`get_jit_stats`).

The new layout keeps implementation details under `compiler/`, `runtime/`, and
`tracing/`, while `_internal` serves as a compatibility layer.

## Migration Notes

- Import paths under `_internal` continue to function, but direct usage should
  migrate to the structured modules under `ember.xcs.compiler` and
  `ember.xcs.runtime`.
- The legacy `ember.xcs._simple` module has been removed; import from
  `jit` from `ember.xcs` instead.
- Invalid configuration values raise `XCSError` rather than `ValueError` to make
  error classes easier to catch.

## Examples

The existing examples in `examples/` continue to run with the new API. Key
patterns:

- `examples/04_compound_ai/simple_ensemble.py`: batching inference with `vmap`.
- `examples/06_performance_optimization/batch_processing.py`: using
  `vmap` + `Config(max_workers=...)` for orchestration.

Run them via `uv run python examples/...` after running `uv sync` as outlined in
`ARCHITECTURE.md`.
