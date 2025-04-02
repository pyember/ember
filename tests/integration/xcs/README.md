# XCS Integration Tests

This directory contains integration tests for the unified XCS architecture. These tests validate the complete, end-to-end functionality of the XCS subsystem across multiple components.

## Test Organization

The tests are organized by architectural component:

- **engine/**: Tests for the unified execution engine
- **graph/**: Tests for the graph representation and dependency analysis
- **jit/**: Tests for Just-In-Time compilation functionality
- **test_unified_architecture.py**: High-level tests of the complete architecture
- **test_jit_ensemble_schedulers.py**: Tests specifically for ensemble workflows with JIT
- **test_jit_strategies_integration.py**: Tests for different JIT strategies

## Test Philosophy

These integration tests focus on validating:

1. **Architecture Correctness**: Ensuring the components work together as designed
2. **End-to-End Functionality**: Testing complete workflows rather than isolated units
3. **Performance Characteristics**: Validating scalability and parallelism
4. **API Stability**: Ensuring public interfaces behave as expected

## Running Tests

```bash
# Run all integration tests
uv run pytest tests/integration/xcs

# Run specific components
uv run pytest tests/integration/xcs/engine
uv run pytest tests/integration/xcs/jit
uv run pytest tests/integration/xcs/graph

# Run with coverage
uv run pytest tests/integration/xcs --cov=src/ember/xcs
```

## Test Design Guidelines

When adding tests to Ember:

1. Focus on testing end-to-end workflows
2. Use realistic operator patterns like those in real applications
3. Validate multiple execution paths and configurations
4. Include explicit assertions about expected behavior
5. Keep tests fast and deterministic
6. Use minimal, self-contained examples that demonstrate functionality
7. Test both happy paths and error handling