# Performance Tests

Tests for measuring JIT optimization performance across different operator patterns.

## Test Patterns

- Linear Chain
- Diamond Pattern 
- Ensemble (width=10)
- Ensemble + Judge
- Nested Ensembles

## Implementations Compared

- No JIT (baseline)
- Regular JIT
- Structural JIT + Sequential Execution
- Structural JIT + Parallel Execution
- Structural JIT + Auto Strategy

## Usage

With pytest:
```bash
# All performance tests
pytest --run-perf-tests tests/integration/performance/

# Specific test file
pytest --run-perf-tests tests/integration/performance/test_nested_jit_performance.py
```

Direct execution:
```bash
# All tests
python tests/integration/performance/test_nested_operators_performance.py --runs 5 --warmup 2

# Specific test
python tests/integration/performance/test_nested_operators_performance.py --test ensemble_judge

# JIT comparison
python tests/integration/performance/test_nested_jit_performance.py
```

## Results

Test results are saved to the `results/` directory as JSON files with timestamps.

## Expected Patterns

- Ensemble operators: Structural JIT + Parallel should be fastest
- Nested patterns: Structural JIT optimizes better than regular JIT
- Diamond patterns: Parallel execution exploits branch parallelism