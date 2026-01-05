# Ember Test Suite

## Philosophy

We follow the testing principles that Google L7+ engineers would apply:
- Test the behavior users rely on, not implementation details
- Fast feedback loops (whole suite < 60 seconds)
- Zero flaky tests - deterministic or properly isolated
- Tests serve as living documentation
- Measure and prevent regressions

## Quick Start

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/           # Fast unit tests (<10ms each)
pytest tests/integration/    # Integration tests (<1s each)  
pytest tests/golden/         # Golden file regression tests

# Run with coverage
pytest --cov=ember --cov-report=html

# Update golden files
UPDATE_GOLDEN=1 pytest tests/golden/

# Run benchmarks
pytest tests/benchmarks/ -v
```

## Test Categories

### Unit Tests (70%)
Fast, focused tests of individual components:
- API contracts
- Algorithm correctness  
- Error handling
- Resource management

### Integration Tests (20%)
Real component interaction tests:
- Provider API calls
- End-to-end workflows
- Thread safety
- Performance characteristics

### Golden Tests (10%)
Regression prevention through versioned outputs:
- Documentation examples
- Common workflows
- API responses

## Key Test Scenarios

### 1. Model API Tests
```python
def test_basic_model_call():
    """What 90% of users will do first."""
    model = Model("gpt-4")
    response = model("What is 2+2?")
    assert "4" in response
```

### 2. Resource Leak Tests
```python
def test_no_memory_leaks():
    """1000 operations shouldn't leak."""
    model = Model("gpt-4")
    assert_no_resource_leaks(
        lambda: model("test"),
        iterations=1000
    )
```

### 3. Concurrency Tests
```python
def test_thread_safe_model_calls():
    """Concurrent usage is safe."""
    model = Model("gpt-4")
    results = run_concurrent(
        lambda i: model(f"Count to {i}"),
        n_threads=10,
        n_calls=100
    )
    assert len(results) == 100
```

## Writing Tests

### Use Test Builders
```python
# Good - explicit and simple
response = model_response_builder(
    content="Test response",
    model="gpt-4",
    prompt_tokens=10
)

# Bad - magic factories
response = ResponseFactory.create()
```

### Test Real Behavior
```python
# Good - tests what users experience
def test_model_timeout():
    model = Model("gpt-4", timeout=1.0)
    with pytest.raises(TimeoutError):
        model("Generate a 10,000 word essay")

# Bad - tests implementation
def test_model_internal_state():
    model = Model("gpt-4")
    assert model._internal_cache == {}
```

### Keep Tests Fast
```python
# Good - mock external calls in unit tests
with mock_provider("openai"):
    model = Model("gpt-4")
    response = model("test")

# Bad - real API calls in unit tests
model = Model("gpt-4")  # Real API call
response = model("test")
```

## Test Utilities

### Resource Monitoring
```python
with monitor_resources() as metrics:
    # Your test code
    pass

assert metrics["memory_delta"] < 10_000_000  # 10MB
assert metrics["fd_delta"] == 0  # No leaked files
```

### Performance Baselines
```python
PerformanceBaseline.measure(
    "model_invocation",
    lambda: model("test"),
    threshold_ms=50  # Fail if >50ms
)
```

### Golden Files
```python
result = model("What is the capital of France?")
assert_matches_golden("france_capital.json", result)
# Run with UPDATE_GOLDEN=1 to update
```

## CI/CD Integration

Tests run automatically on:
- Every commit (unit tests)
- Every PR (full suite)
- Nightly (integration tests with real APIs)
- Release (performance benchmarks)

## Debugging Test Failures

### Flaky Tests
1. Check for race conditions
2. Add deterministic seeds
3. Isolate shared state
4. Use proper synchronization

### Performance Regressions
1. Check recent commits
2. Profile with `pytest --profile`
3. Compare against baselines
4. Look for O(n²) algorithms

### Resource Leaks
1. Run with `pytest --monitor-resources`
2. Check file descriptor counts
3. Monitor memory growth
4. Use context managers

## Test Maintenance

### Weekly Tasks
- Review and fix any flaky tests
- Update golden files if needed
- Check performance trends
- Prune obsolete tests

### Monthly Tasks  
- Review test coverage gaps
- Update provider mocks
- Benchmark against competition
- Documentation accuracy check

## Contributing Tests

When adding new tests:
1. **Follow the pattern** - Look at existing tests
2. **Test user behavior** - Not implementation
3. **Keep it fast** - <10ms for unit tests
4. **Make it deterministic** - No random failures
5. **Document why** - Explain what you're testing

Remember: Tests are code too. They need the same care and attention as production code.