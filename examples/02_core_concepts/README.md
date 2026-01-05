# Core Concepts

Learn fundamental concepts for building robust AI applications with Ember's simplified API. These examples cover the core building blocks that make AI systems reliable, maintainable, and production-ready.

## What You'll Learn

- **Function-based operators** - Any function can be an operator, no base classes needed
- **Rich data validation** - Type-safe specifications with EmberModel (Pydantic)
- **State management** - Centralized configuration and resource access through context
- **Error resilience** - Robust error handling, retries, and fallback patterns
- **Type safety** - Leverage Python's type system for reliable applications

## Examples

### [operators_basics.py](./operators_basics.py)
**Beginner | ~8 minutes**

Master Ember's function-based operator approach with `@jit` optimization, composition patterns, and batch processing.

### [rich_specifications.py](./rich_specifications.py)
**Intermediate | ~10 minutes**

Build type-safe operators using EmberModel for structured validation, custom validators, and error handling.

### [context_management.py](./context_management.py)
**Intermediate | ~8 minutes**

Manage application state, configuration, and resources through Ember's context system with isolation and lifecycle management.

### [error_handling.py](./error_handling.py)
**Intermediate | ~10 minutes**

Create resilient applications with retry strategies, fallback chains, circuit breakers, and production error patterns.

### [type_safety.py](./type_safety.py)
**Beginner | ~7 minutes**

Use Python type hints, dataclasses, protocols, and runtime validation for maintainable AI applications.

## Learning Workflows

### Quick Start (20 minutes)
1. **operators_basics.py** - Foundation concepts
2. **rich_specifications.py** - Data validation
3. **error_handling.py** - Resilience patterns

### Comprehensive Study (45 minutes)
1. **operators_basics.py** - Operator fundamentals
2. **type_safety.py** - Type system integration  
4. **context_management.py** - State management
3. **rich_specifications.py** - Advanced validation
5. **error_handling.py** - Production resilience

## Getting Started

**Prerequisites:** Complete `01_getting_started`, basic Python knowledge

**Setup:** Examples replay recorded output by default (no API calls).

```bash
cd examples/02_core_concepts
uv run python operators_basics.py
```

## Next Steps

- **[03_simplified_apis](../03_simplified_apis/README.md)** - High-level APIs
- **[04_compound_ai](../04_compound_ai/)** - Multi-component systems
- **[Examples Index](../README.md)** - All examples
