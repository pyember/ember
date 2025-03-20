# Ember Operator Examples

This directory contains examples demonstrating how to create, compose, and use Ember operators for building AI systems.

## Examples

- `minimal_operator_example.py` - Basic operator creation and usage
- `composition_example.py` - Composing multiple operators together
- `container_operator_example.py` - Using container operators for complex logic
- `diverse_ensemble_operator_example.py` - Building ensembles of different models
- `container_simplified.py` - Simplified container operator pattern
- `custom_prompt_example_caravan.py` - Using custom prompts with operators

## Running Examples

To run any example, use the following command format:

```bash
# Using uv (recommended)
uv run python src/ember/examples/operators/example_name.py

# Or if in an activated virtual environment
python src/ember/examples/operators/example_name.py
```

Replace `example_name.py` with the desired example file.

## Operator Concepts

Ember operators are the fundamental building blocks for AI systems, similar to PyTorch's `nn.Module`. They provide:

- Type-safe input/output interfaces
- Specification-driven execution
- Support for composition and nesting
- Automatic optimization via XCS

## Next Steps

After mastering operators, explore:

- `xcs/` - For advanced execution optimization
- `advanced/` - For complex operator patterns and architectures
