# Ember XCS Examples

This directory contains examples for the XCS (eXecutable Computation System) capabilities of Ember, which provide performance optimization and advanced execution features.

## Examples

- `jit_example.py` - Using the JIT compilation system
- `enhanced_jit_example.py` - Advanced JIT features
- `auto_graph_example.py` - Automatic execution graph building
- `auto_graph_simplified.py` - Simplified autograph API
- `simple_autograph_example.py` - Basic autograph usage
- `test_xcs_implementation.py` - Testing XCS core functionality
- `example_simplified_xcs.py` - Simplified XCS API examples

## Running Examples

To run any example, use the following command format:

```bash
poetry run python src/ember/examples/xcs/example_name.py
```

Replace `example_name.py` with the desired example file.

## XCS Concepts

The XCS system provides several key capabilities:

- Just-In-Time (JIT) compilation for operators
- Automatic execution graph building and optimization
- Parallel execution scheduling
- Function transformation (vmap, pmap, etc.)

## Next Steps

After learning about XCS, explore:

- `advanced/` - For complex systems using XCS features
- `integration/` - For examples of integrating XCS with other systems
