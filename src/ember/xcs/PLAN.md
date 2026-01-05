# XCS Refactor TODO

Goal: Rebuild the `ember.xcs` package with a clean structure, Google Python Style Guide compliant code, and clearer separation between public APIs, compiler analysis, runtime execution, and utilities.

## Target Package Layout
```
ember/xcs/
├── __init__.py            # re-export public API
├── README.md              # user guide (refresh)
├── PLAN.md                # this plan
├── errors.py              # shared XCS exceptions
├── config.py              # public configuration types
├── api/
│   ├── __init__.py        # jit/vmap/pmap/scan/grad exports
│   ├── jit.py             # decorator implementation and stats helpers
│   ├── transforms.py      # transformation façade (vmap/pmap/scan/grad)
│   └── stats.py           # profiling/statistics accessors
├── compiler/
│   ├── __init__.py
│   ├── analysis.py        # function/op analysis helpers
│   ├── builder.py         # IR builder from traces
│   ├── graph.py           # IR data structures
│   └── parallelism.py     # parallelism discovery
├── runtime/
│   ├── __init__.py
│   ├── engine.py          # execution engine and context
│   └── profiler.py        # profiling implementation
├── tracing/
│   ├── __init__.py
│   └── python_tracer.py   # sys.settrace tracer
└── utils/
    ├── __init__.py
    └── pytree.py          # pytree interop helpers
```

Legacy `_internal` modules will remain as thin forwarders to the new structure to avoid breaking imports in docs/tests.

## File-by-File Tasks

- [x] `__init__.py` — re-export updated API and refresh docstring.
- [x] `errors.py` — centralized `XCSError`.
- [x] `config.py` — rebuilt dataclass with validation + presets.
- [x] `api/jit.py` — new decorator, profiling hook, and `get_jit_stats`.
- [x] `api/transforms.py` — rebuilt transforms with orchestration-aware batching.
- [x] `api/stats.py` — helper accessors for profiling.
- [x] `compiler/analysis.py` — refreshed AST heuristics & helpers.
- [x] `compiler/graph.py` — new IR data classes.
- [x] `compiler/builder.py` — runtime trace to IR graph.
- [x] `compiler/parallelism.py` — graph analysis + speedup heuristics.
- [x] `runtime/engine.py` — sequential/parallel executor.
- [x] `runtime/profiler.py` — new profiling primitives.
- `tracing/python_tracer.py`
  - Sys.settrace-based tracer with safe locking.
- [x] `utils/pytree.py` — pytree wrappers + registration.
- [x] `_internal/*` — compatibility shims pointing to new modules.
- [x] `README.md` — documented new layout and usage.
- [x] `examples/` — verified existing samples still apply.
- [ ] Tests/docs — attempted `pytest tests/unit/xcs`, blocked by existing indentation error in `src/ember/models/pricing_updater.py`.

## Execution Order

1. Stand up shared foundations (`errors`, `config`, `utils/pytree`).
2. Implement compiler layer (analysis, graph, builder, parallelism, tracing).
3. Implement runtime layer (engine, profiler).
4. Implement API layer (jit, transforms, stats) and update `__init__`/README.
5. Wire compatibility shims, run focused tests, and polish docs/examples.

Progress will be tracked by checking off each bullet as files are implemented.
