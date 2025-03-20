# Ember Framework Testing

This directory contains tests for the Ember framework. The test suite is designed to validate the functionality, reliability, and performance of Ember's components.

## Testing Philosophy

Our testing approach follows these key principles:

1. **Avoid overmocking**: Test actual implementations rather than creating duplicate mock versions of core system components. Only mock external dependencies and boundaries.

2. **Test real behavior**: Tests should validate how your code actually works, not how you think it works. Avoid parallel implementations in test code.

3. **Root cause debugging**: Fix issues at the source rather than working around them in tests. When tests fail, understand and address the underlying problem, not just the symptom.

4. **Integration testing**: Include tests that verify how components work together in the real system, not just in isolation.

5. **Minimal test doubles**: Create the minimal test double needed for the test, preserving real behavior whenever possible.

6. **Targeted mocking**: Mock at the level of the dependency being replaced, not at the level of the entire subsystem.

7. **Test for correctness**: Focus tests on behavior correctness rather than implementation details that might change.

## Test Structure

The test directory is organized as follows:

- **unit/**: Unit tests for individual components
- **integration/**: Integration tests for component interactions
- **helpers/**: Test utilities and fixtures
- **fuzzing/**: Fuzz testing for robustness

## Test Doubles

We provide several types of test doubles to assist in testing:

### Minimal Test Doubles

Located in `helpers/` with the `*_minimal_doubles.py` naming pattern. These implement just enough functionality to test client code without duplicating the entire implementation.

- `xcs_minimal_doubles.py`: Minimal doubles for XCS components
- `operator_minimal_doubles.py`: Minimal doubles for operators
- `model_minimal_doubles.py`: Minimal doubles for model registry

### Test Fixtures

Located in `helpers/` directory. These provide standardized test data and configurations.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit

# Run integration tests only (requires RUN_INTEGRATION_TESTS=1)
RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration

# Run with coverage
uv run pytest --cov=src/ember

# Or using uvx shorthand for running tools directly
uvx pytest tests/unit
```

## Integration Testing

Integration tests verify that components work correctly together. We have several integration test suites:

1. **Provider Discovery**: Tests model provider discovery across OpenAI, Anthropic, etc. (`tests/integration/core/registry/test_provider_discovery.py`)

2. **Operator Composition**: Tests complex operator compositions with different patterns (`tests/integration/core/test_operator_integration.py`)

3. **XCS Execution**: Tests the XCS tracing and execution pipeline (`tests/integration/xcs/test_xcs_integration.py`)

4. **Data Processing**: Tests end-to-end data pipelines (`tests/integration/core/utils/data/test_data_end_to_end.py`)

## Simplified Import Structure

We've simplified the import structure to make it more intuitive:

```python
# Import the Operator base class
from ember.operator import Operator

# Import NON components
from ember.non import UniformEnsemble, JudgeSynthesis, Sequential

# Use them together
ensemble = UniformEnsemble(num_units=3, model_name="openai:gpt-4o")
judge = JudgeSynthesis(model_name="anthropic:claude-3-opus")
pipeline = Sequential(operators=[ensemble, judge])
```

## Best Practices

1. **Use minimal test doubles**: Prefer the minimal test doubles in `helpers/*_minimal_doubles.py` over complex mocks.

2. **Avoid monkeypatching**: Design classes to accept dependencies through constructor injection rather than monkeypatching.

3. **Test through public APIs**: Focus on testing public interfaces rather than internal implementation details.

4. **Verify behavior, not implementation**: Assert on outputs and state changes rather than method calls.

5. **Integration test coverage**: Ensure critical workflows have integration tests that verify components work together correctly.
