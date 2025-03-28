# Ember Examples

This directory contains example code demonstrating the Ember framework capabilities, from simple examples to advanced use cases.

## Getting Started

If you're new to Ember, start with the basic examples that demonstrate core concepts:

```bash
# Run a minimal example
uv run python src/ember/examples/basic/minimal_example.py

# Try a simple model example
uv run python src/ember/examples/models/model_registry_example.py

# Or using the uvx shorthand for running tools
uvx pytest tests/unit/  # Run unit tests
```

## Directory Structure

- **[basic/](./basic/)** - Simple examples for beginners to get started
- **[models/](./models/)** - Examples for working with LLM models and the model registry
- **[operators/](./operators/)** - Examples showing how to create and compose operators
- **[data/](./data/)** - Examples for data loading, transformation, and dataset usage
- **[xcs/](./xcs/)** - Examples for using the XCS (Accelerated Compound Systems) features
- **[advanced/](./advanced/)** - Advanced examples with complex workflows and optimizations
- **[integration/](./integration/)** - Examples showing integration with other systems

## Example Difficulty Levels

Examples are categorized by difficulty level in their docstrings:

- **Basic**: Fundamental concepts and minimal working examples
- **Intermediate**: More complex but common use cases
- **Advanced**: Complex workflows and specialized features

## Popular Examples

- [Minimal Example](./basic/minimal_example.py) - The simplest possible Ember operator
- [Model Registry Example](./models/model_registry_example.py) - Working with LLM models
- [JIT Example](./xcs/jit_example.py) - Using the JIT system for performance optimization
- [Ensemble Operator Example](./operators/ensemble_operator_example.py) - Creating multi-model ensembles
- [Data API Example](./data/data_api_example.py) - Working with datasets

## Complete Application Examples

For complete application examples that show how to build real-world AI systems with Ember, see:

- [Advanced Reasoning System](./advanced/reasoning_system.py)
- [Parallel Pipeline Example](./advanced/parallel_pipeline_example.py)
- [Auto Graph Simplified](./xcs/auto_graph_simplified.py)

## Documentation

For more information, see the [project README](../../../../README.md) and documentation files in the [docs/](../../../../docs/) directory.
