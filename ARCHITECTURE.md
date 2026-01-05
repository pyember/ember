# Ember Architecture

## Design Philosophy

Ember follows three core principles:

1. **Simple by Default** - Zero configuration, direct function calls, no boilerplate
2. **Progressive Disclosure** - Advanced features available when needed, hidden when not
3. **10x Performance** - Automatic optimization without manual tuning

## System Overview

```
┌────────────────────────────────────────────────────────────┐
│                         PUBLIC API                         │
├─────────────────┬─────────────────┬────────────────────────┤
│     models()    │   operators()   │    data.stream()       │
│  Direct LLM API │ Composable AI   │  Streaming Data        │
└────────┬────────┴────────┬────────┴───────┬────────────────┘
         │                 │                │
┌────────▼────────┬────────▼────────┬───────▼───────────────┐
│  Model Registry │ Operator System │   Data Pipeline       │
│                 │                 │                       │
│  • Provider     │ • Composition   │  • Loaders            │
│    Resolution   │ • Validation    │  • Transformers       │
│  • Cost Tracking│ • JAX Pytrees   │  • Samplers           │
└────────┬────────┴────────┬────────┴───────┬───────────────┘
         │                 │                │
┌────────▼─────────────────▼────────────────▼─────────────────┐
│                      XCS ENGINE                             │
│                                                             │
│  • Automatic JIT Compilation                                │
│  • Parallelism Detection                                    │
│  • Execution Optimization                                   │
└─────────────────────────────────────────────────────────────┘
```

### CLI System Integration

The CLI provides comprehensive management and introspection capabilities:

```bash
# Interactive setup with provider selection
ember setup  # Launches React-based wizard

# Configuration management
ember configure get KEY           # Get config value
ember configure set KEY VALUE     # Set config value  
ember configure list             # Show all config
ember configure show SECTION     # Show section

# Context introspection
ember context view               # View resolved configuration
ember context view --format json # Output as JSON
ember context view --filter models # Filter to specific path
ember context validate           # Validate configuration

# Registry introspection
ember registry list-models       # List available models
ember registry list-models --provider openai # Filter by provider
ember registry list-models --verbose # Detailed information
ember registry list-providers    # Show provider status
ember registry info gpt-4        # Detailed model info

# Testing and discovery
ember test [--model MODEL]       # Test connection
```

The setup wizard (`@ember-ai/setup`) features:
- React/Ink-based interactive UI
- Provider-specific API key validation
- Automatic configuration file creation
- Integration with EmberContext for credential storage

## Core Components

### 1. Context System

Centralized configuration and credential management with thread-safe and async-safe isolation.

The context system provides a simplified, unified API following the principle of "one obvious way":

```python
from ember import context

# Primary API - simple and clear
ctx = context.get()  # Get current context (creates if needed)

# Context manager for temporary overrides
with context.manager(models={"default": "gpt-4", "temperature": 0.9}) as ctx:
    # All operations in this block use these settings
    response = models("Hello")  # Uses gpt-4 with temperature 0.9
    
# Original context automatically restored

# Direct configuration access
from ember.context import get_config, set_config
default_model = get_config("models.default", "gpt-3.5-turbo")
set_config("models.temperature", 0.7)
```

**Key Features:**
- **Thread-safe and async-safe**: Context propagates correctly across boundaries
- **Hierarchical configuration**: Child contexts inherit from parents
- **Multiple credential sources**: Runtime > Environment > Config file > Defaults
- **Clean scoping**: Context managers ensure proper cleanup

**Configuration Priority:**
1. Runtime context (highest)
2. Environment variables
3. ~/.ember/config.yaml
4. Internal defaults (lowest)

**Async Context Propagation:**

The context system uses Python's `contextvars` for proper async isolation:

```python
import asyncio
from ember.context import get_context, create_context

async def process_with_model(text: str) -> str:
    # Context automatically propagates across async boundaries
    ctx = get_context()  # Gets the correct context
    model = ctx.get_config("models.default")
    print(f"Task {asyncio.current_task().get_name()}: Using {model}")
    
    # Context remains isolated even with concurrent execution
    await asyncio.sleep(0.1)
    return f"Processed '{text}' with {model}"

async def main():
    # Each task gets its own isolated context
    async def task_with_context(name: str, model: str, text: str):
        with create_context(models={"default": model}):
            return await process_with_model(text)
    
    # Run tasks concurrently with different contexts
    results = await asyncio.gather(
        task_with_context("task1", "gpt-4", "Hello"),
        task_with_context("task2", "claude-3-opus", "World"),
        task_with_context("task3", "gemini-1.5-pro-latest", "Async"),
    )
    
    for result in results:
        print(result)
    # Output:
    # Task task-1: Using gpt-4
    # Task task-2: Using claude-3-opus  
    # Task task-3: Using gemini-1.5-pro-latest
    # Processed 'Hello' with gpt-4
    # Processed 'World' with claude-3-opus
    # Processed 'Async' with gemini-1.5-pro-latest
```

**Thread-Safe Context Usage:**

The context system uses thread-local storage with proper locking:

```python
import threading
from ember.context import get_context, create_context

def worker(name: str, model: str):
    # Each thread gets isolated context
    with create_context(models={"default": model}):
        ctx = get_context()
        print(f"{name}: Using {ctx.get_config('models.default')}")
        # Do work with thread-local model

# Launch workers with different models
threads = [
    threading.Thread(target=worker, args=("Worker1", "gpt-4")),
    threading.Thread(target=worker, args=("Worker2", "claude-3")),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

**Configuration Management:**

```python
from ember.context import get_context, with_context
from ember.api import models

# Get and modify configuration
ctx = get_context()
ctx.set_config("models.temperature", 0.7)
ctx.save()  # Persist to disk

# Temporary configuration overrides
with with_context(models={"temperature": 0.9, "max_tokens": 2000}):
    # This block uses high temperature and more tokens
    response = models("gpt-4", "Write a creative story")
    
# Back to original settings
response = models("gpt-4", "Summarize this document")  # Uses temperature=0.7
```

### 2. Credential Management

The context system provides secure, hierarchical credential management:

```python
from ember import context

ctx = context.get()

# Credential lookup hierarchy (first found wins):
# 1. Runtime context credentials
# 2. Environment variables (OPENAI_API_KEY, etc.)
# 3. Config file (~/.ember/config.yaml)
# 4. Separate credentials file (~/.ember/credentials.yaml)

# Get credential with fallback chain
api_key = ctx.get_credential("openai", "OPENAI_API_KEY")

# Set credentials at runtime (not persisted)
with context.manager(credentials={"openai_api_key": "sk-temp-key"}):
    # This block uses the temporary key
    response = models("gpt-4", "Hello")

# Persist credentials securely
ctx.set_config("credentials.openai_api_key", "sk-prod-key")
ctx.save()  # Saves to ~/.ember/config.yaml with proper permissions
```

**Security Features:**
- Credentials never logged or displayed in errors
- Config files created with 0600 permissions (user-only read/write)
- Support for credential rotation without code changes
- Clear error messages that don't expose sensitive data

### 3. Models API

Direct LLM invocation without client initialization:

```python
# Simple case - no setup required
response = models("gpt-4", "Hello world")

# Advanced case - reusable configuration
gpt4 = models.instance("gpt-4", temperature=0.7)
```

**Design Decisions:**
- Registry pattern for provider management
- Lazy initialization for fast startup
- Integrated cost and usage tracking
- Thread-safe singleton implementation

### 2. Operators System

Composable building blocks for AI applications:

```python
# Function decorator approach
@operators.op
def summarize(text: str) -> str:
    return models("gpt-4", f"Summarize: {text}")

# Class-based with validation
class ValidatedOp(Operator):
    input_spec = InputModel
    output_spec = OutputModel
```

**Design Decisions:**
- JAX pytree compatibility for automatic differentiation
- Optional but powerful validation system
- Composition over inheritance
- Zero overhead when validation not used

### 3. Data Pipeline

Streaming-first data loading with progressive enhancement:

```python
# Basic streaming
for item in stream("dataset"):
    process(item)

# Chained transformations
stream("dataset").filter(valid).transform(clean).batch(32)
```

**Design Decisions:**
- Memory-efficient streaming by default
- Explicit materialization when needed
- Protocol-based extensibility
- Built-in caching layer

### 4. XCS Execution Engine

Zero-configuration optimization:

```python
@xcs.jit
def complex_workflow(data):
    # Automatically optimized
    return pipeline(data)

# Automatic parallelization
results = xcs.vmap(process)(batch)
```

**Design Decisions:**
- Tracing-based optimization
- Automatic strategy selection
- JAX backend for numerical operations
- Orchestration for I/O-bound tasks

## Key Design Patterns

### Registry Pattern
Single source of truth for each resource type:
- `ModelRegistry` - LLM provider management
- `DataRegistry` - Dataset loader management
- Thread-safe with proven single-lock approach

### Module System
JAX-compatible base class for all operators:
- Automatic parameter detection
- Pytree registration
- Clean composition semantics

### Progressive Disclosure
Three levels of API complexity:
1. **Simple Functions** - Direct calls for basic use
2. **Decorators** - Enhancement without boilerplate
3. **Classes** - Full control when needed

### Type-Driven Development
Optional but powerful when used:
- Pydantic models for validation
- Type hints guide system behavior
- Runtime validation from static types

## Performance Architecture

### JIT Compilation Strategy
1. Function decorated with `@jit`
2. First call traces execution
3. IR built from trace
4. Optimal backend selected:
   - JAX for numerical operations
   - Async orchestration for I/O
5. Subsequent calls use compiled version

### Parallelization Detection
- Automatic detection of map operations
- Data dependency analysis
- Optimal chunking for throughput
- Zero configuration required

### Memory Management
- Streaming by default for data
- Lazy evaluation where possible
- Explicit materialization points
- Automatic garbage collection hints

## Extension Points

### Adding Providers
1. Implement `BaseProvider` interface
2. Register with `ModelRegistry`
3. No core changes required

### Custom Operators
1. Inherit from `Operator` base
2. Define `call()` method
3. Optional validation specs
4. Automatic JAX integration

### Data Loaders
1. Implement loader protocol
2. Register with `DataRegistry`
3. Streaming support automatic

## Testing Strategy

### Unit Tests
- Isolated component testing
- Minimal test doubles
- Type testing utilities

### Integration Tests
- Cross-module interactions
- Real provider testing
- Performance benchmarks

### Test Patterns
- Helper modules for common setups
- Simplified imports for isolation
- Comprehensive coverage tracking

## Security Considerations

### API Key Management
- Environment variable loading
- No keys in code
- Secure defaults

### Input Validation
- Optional but recommended
- Pydantic integration
- Type-safe boundaries

### Rate Limiting
- Provider-level handling
- Automatic retry logic
- Exponential backoff

## Future Architecture

### Planned Enhancements
1. **Distributed Execution** - Multi-node XCS
2. **Model Quantization** - Automatic optimization
3. **Streaming Inference** - Token-level streaming
4. **Edge Deployment** - Browser/mobile runtime

### Design Principles Maintained
- Simple API remains simple
- Advanced features stay optional
- Performance improvements automatic
- Backward compatibility preserved

## Architecture Decision Records

### ADR-001: Direct Instantiation
**Decision**: Use direct function calls instead of dependency injection
**Rationale**: Eliminates boilerplate, improves discoverability
**Consequences**: Simpler API, easier testing, less flexibility

### ADR-002: Registry Pattern
**Decision**: Single registry per resource type
**Rationale**: Clear ownership, thread-safe, extensible
**Consequences**: Centralized management, potential bottleneck

### ADR-003: JAX Integration
**Decision**: Base operators on JAX pytrees
**Rationale**: Automatic differentiation, JIT compilation
**Consequences**: Power user features, slight complexity

### ADR-004: Streaming First
**Decision**: Default to streaming for data operations
**Rationale**: Memory efficiency, scalability
**Consequences**: Explicit materialization needed sometimes

This architecture embodies the principles of simplicity, performance, and extensibility that guide Ember's development.
