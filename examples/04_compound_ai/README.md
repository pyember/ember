# Compound AI Systems

Learn to build sophisticated AI systems by composing multiple models, operators, and processing stages. These examples demonstrate ensemble patterns, judges, synthesis, and progressive complexity using Ember's new model primitives.

## Examples

### [simple_ensemble.py](./simple_ensemble.py) | ~5 min
Build ensemble systems using ModelText operators, custom voting strategies, and batch processing with `@jit` optimization.

### [operators_progressive_disclosure.py](./operators_progressive_disclosure.py) | ~15 min  
Explore Ember's 5-level operator complexity: from simple `@op` functions to JAX-integrated learnable systems.

### [judge_synthesis.py](./judge_synthesis.py) | ~12 min
Implement judge systems for response evaluation and synthesis patterns for combining multiple AI perspectives.

### [specifications_progressive.py](./specifications_progressive.py) | ~10 min
Master progressive specification complexity: from no validation to rich EmberModel validation with prompt templates.

## Learning Workflows

**Quick Start (15 min):** `simple_ensemble.py` → `judge_synthesis.py`

**Model Primitives (25 min):** All examples focusing on ModelCall/ModelText usage

**Advanced Patterns (40 min):** All examples in order for complete system patterns

## Getting Started

```bash
# Examples replay recorded output (no API keys required)
python examples/04_compound_ai/simple_ensemble.py
```

## Key Concepts

- **ModelCall/ModelText**: New model primitives for response handling
- **Ensemble/Chain**: Operator composition patterns  
- **Progressive disclosure**: Scale complexity only when needed
- **Judge systems**: Automated response evaluation and selection
- **Synthesis**: Combining multiple perspectives coherently

## Next Steps

- **[05_data_processing](../05_data_processing/)** - Advanced data workflows
- **[Examples Index](../README.md)** - All examples
