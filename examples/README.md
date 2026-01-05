# Ember Examples

Practical examples demonstrating Ember's capabilities, organized from beginner to advanced.

## Quick Start

```bash
# Verify installation (no API keys required)
python examples/01_getting_started/hello_world.py

# Make your first model call (requires API key)
ember setup  # Configure credentials first
python examples/01_getting_started/first_model_call.py
```

## Directory Structure

The examples follow a progressive learning path:

| Directory | Focus | API Keys Required |
|-----------|-------|-------------------|
| `01_getting_started/` | Installation, first calls, basics | Partial |
| `02_core_concepts/` | Operators, types, composition | Yes |
| `03_simplified_apis/` | High-level ergonomic APIs | Yes |
| `04_compound_ai/` | Multi-model workflows | Yes |
| `05_data_processing/` | Datasets and streaming | Partial |
| `06_performance_optimization/` | JIT, batching, speed | Yes |
| `07_error_handling/` | Robust production patterns | Yes |
| `08_advanced_patterns/` | XCS transforms, advanced composition | Yes |
| `09_practical_patterns/` | RAG, CoT, structured output | Yes |
| `10_evaluation_suite/` | Benchmarking and testing | Yes |

Each directory contains a `README.md` with detailed explanations.

## Prerequisites

1. **Python 3.11+** installed
2. **Ember** installed: `pip install ember-ai` or `uv sync` from source
3. **API credentials** for examples requiring model calls:
   ```bash
   ember setup  # Interactive wizard
   # Or manually:
   ember configure set providers.openai.api_key "sk-..."
   ```

## Recommended Learning Path

**New to Ember:**
1. `01_getting_started/hello_world.py` - Verify installation
2. `01_getting_started/first_model_call.py` - Basic model usage
3. `02_core_concepts/operators_basics.py` - Understand operators

**Building Applications:**
1. `09_practical_patterns/` - RAG, chain-of-thought, structured output
2. `07_error_handling/` - Production robustness

**Optimizing Performance:**
1. `06_performance_optimization/jit_basics.py` - JIT compilation
2. `06_performance_optimization/batch_processing.py` - Throughput optimization

**Advanced Usage:**
1. `08_advanced_patterns/` - XCS transforms
2. `10_evaluation_suite/` - Systematic evaluation

## Running Tests

```bash
uv run pytest tests/examples -q
```

## Contributing Examples

When adding new examples:
1. Place in the appropriate numbered directory
2. Include a docstring explaining what the example demonstrates
3. Handle missing API keys gracefully
4. Update the directory's README.md
