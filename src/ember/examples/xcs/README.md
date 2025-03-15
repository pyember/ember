# Ember XCS Examples

This directory contains examples demonstrating the XCS (Accelerated Compound Systems execution engine) capabilities of Ember, which provide performance optimization and advanced execution features.

## Example Guide by Concept

### JIT Compilation Systems

Ember provides three complementary approaches to JIT compilation:

1. **jit** - Trace-based JIT compilation
   - `jit_example.py` - Demonstrates the `@jit` decorator with performance comparisons
   - Shows how tracing and caching work for optimizing operator execution

2. **structural_jit** - Structure-based JIT compilation
   - `enhanced_jit_example.py` - Demonstrates hierarchical operator analysis and optimization

3. **autograph** - Explicit graph building
   - `auto_graph_example.py` - Shows manual graph building with LLM examples
   - `auto_graph_simplified.py` - Simplified version without LLM dependencies
   - `simple_autograph_example.py` - Basic introduction to the autograph context manager

### General Examples

- `test_xcs_implementation.py` - Testing XCS core functionality
- `example_simplified_xcs.py` - Simplified XCS API examples

## Choosing the Right Example

- **New to XCS?** Start with `jit_example.py` for a basic introduction
- **Want to understand JIT vs. structural_jit?** See `enhanced_jit_example.py`
- **Need manual graph control?** Check out `auto_graph_simplified.py`
- **Building complex operator compositions?** The hierarchical analysis in `enhanced_jit_example.py` is most relevant

## Running Examples

To run any example, use the following command format:

```bash
poetry run python src/ember/examples/xcs/example_name.py
```

Replace `example_name.py` with the desired example file.

## XCS Core Concepts

The XCS system provides several key capabilities:

- **Just-In-Time Compilation**: Three approaches for different use cases
  - Trace-based JIT: Analyzes actual execution patterns
  - Structural JIT: Examines operator composition without tracing
  - Autograph: Explicit graph construction and execution

- **Execution Optimization**:
  - Automatic parallelization of independent operations
  - Intelligent scheduling based on dependencies
  - Caching of execution plans for repeated use

- **Function Transformations**:
  - vmap: Vectorization of operations
  - pmap: Parallelization across worker threads
  - mesh_sharded: Distribution across device meshes

For a comprehensive explanation of the relationships between jit, structural_jit, and autograph, see `docs/xcs/JIT_OVERVIEW.md`.

## Next Steps

After exploring these examples, consider:

- `advanced/` - For complex systems using XCS features
- `integration/` - For examples of integrating XCS with other systems
