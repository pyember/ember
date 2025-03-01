# Ember Architecture

This document provides a comprehensive overview of Ember's architecture, its core components, and the design principles that guide the project.

## System Overview

Ember is organized around a modular, extensible architecture that emphasizes:

1. **Composability**: Components can be combined and nested to create complex AI systems
2. **Extensibility**: All major subsystems support custom extensions via registries
3. **Testability**: Dependency injection enables comprehensive testing
4. **Performance**: Optional graph-based execution for parallel operations

The diagram below illustrates the high-level architecture:

```
┌────────────────────────────────────────────┐
│               Application                  │
├────────────────────────────────────────────┤
│                                            │
│  ┌────────────┐    ┌────────────────────┐  │
│  │            │    │                    │  │
│  │ EmberApp   │◄───┤ ConfigManager      │  │
│  │ Context    │    │                    │  │
│  │            │    └────────────────────┘  │
│  │            │                            │
│  │            │    ┌────────────────────┐  │
│  │            │◄───┤ ModelRegistry      │  │
│  │            │    │                    │  │
│  │            │    └────────────────────┘  │
│  │            │                            │
│  │            │    ┌────────────────────┐  │
│  │            │◄───┤ OperatorRegistry   │  │
│  │            │    │                    │  │
│  │            │    └────────────────────┘  │
│  │            │                            │
│  └────────────┘    ┌────────────────────┐  │
│                    │                    │  │
│                    │ UsageService       │  │
│                    │                    │  │
│                    └────────────────────┘  │
│                                            │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│                Execution                    │
├────────────────────────────────────────────┤
│                                            │
│  ┌────────────┐    ┌────────────────────┐  │
│  │            │    │                    │  │
│  │ XCSGraph   │◄───┤ Operators          │  │
│  │            │    │                    │  │
│  └─────┬──────┘    └────────────────────┘  │
│        │                                    │
│        ▼                                    │
│  ┌────────────┐    ┌────────────────────┐  │
│  │            │    │                    │  │
│  │ Execution  │◄───┤ Schedulers         │  │
│  │ Plan       │    │                    │  │
│  │            │    └────────────────────┘  │
│  └────────────┘                            │
│                                            │
└────────────────────────────────────────────┘
```

## Core Components

### EmberAppContext

The `EmberAppContext` serves as the central access point and dependency injection container for Ember applications. It:

- Manages configuration and provides access to services
- Instantiates and configures registries
- Creates consistent logger instances
- Enables easy testing through dependency injection

```python
# Example of accessing the application context
from ember.core.app_context import get_app_context

context = get_app_context()
model_service = context.model_service
config = context.config_manager.get_config("model_registry")
```

### Configuration System

The configuration system provides a unified approach to managing settings across Ember:

- **ConfigManager**: Loads, merges, and caches configuration files
- **Environment Variable Resolution**: Automatically resolves `${ENV_VAR}` placeholders
- **Configuration Hierarchy**: Supports base configs with specific overrides

Configuration files are YAML-based and located in the relevant component's directory structure.

### Registry System

Ember uses registries to manage the various components that can be plugged into the system:

#### ModelRegistry

The `ModelRegistry` provides a centralized repository for accessing language models:

- **Discovery**: Auto-discovers model configurations from YAML files
- **Registration**: Allows dynamic registration of model metadata
- **Instantiation**: Creates provider-specific model instances through the `ModelFactory`
- **Lookup**: Retrieves models by ID or enum

#### OperatorRegistry

The `OperatorRegistry` manages the collection of available operators:

- **Standard Operators**: Built-in operators for common tasks (ensemble, judge, etc.)
- **Custom Operators**: User-defined operators can be registered
- **Lookup**: Provides access to operator implementations

### Model System

The model system handles the integration with various AI model providers:

- **Provider Implementations**: Support for OpenAI, Anthropic, DeepMind, etc.
- **BaseProviderModel**: Common interface across all providers
- **ModelService**: High-level facade for model invocation
- **UsageService**: Tracks API usage, cost, and rate limits

### Operator System

Operators are the core computational units in Ember:

- **Base Operator**: Abstract base class with forward() method
- **Signature**: Type definitions for inputs and outputs
- **Composition**: Operators can contain and invoke other operators
- **Concurrency**: Operators can specify execution plans for parallel processing

### Execution Engine (XCS)

The XCS (eXecution Control System) handles graph-based execution:

- **XCSGraph**: Directed acyclic graph representing operator dependencies
- **XCSNode**: Individual nodes in the graph, containing operators
- **Scheduler**: Manages execution order and parallelization
- **Tracer**: Records execution details for debugging and optimization

## Package Structure

| Package | Description |
|---------|-------------|
| `ember.core` | Core framework components |
| `ember.core.app_context` | Application context and dependency injection |
| `ember.core.configs` | Configuration management |
| `ember.core.types` | Type system and protocols |
| `ember.core.registry.model` | Model registry and provider implementations |
| `ember.core.registry.operator` | Operator registry and base classes |
| `ember.core.registry.prompt_signature` | Signature definitions for typed I/O |
| `ember.core.utils` | Utility functions and classes |
| `ember.core.non` | High-level Network of Networks operators |
| `ember.xcs` | Execution engine components |
| `ember.xcs.graph` | Graph definition and manipulation |
| `ember.xcs.engine` | Execution scheduling and optimization |
| `ember.xcs.tracer` | Execution tracing and profiling |
| `ember.examples` | Example applications and usage patterns |

## Design Patterns

Ember employs several design patterns to maintain a clean and extensible architecture:

1. **Registry Pattern**: Central repositories for discoverable components
2. **Factory Pattern**: Creation of provider-specific instances
3. **Dependency Injection**: Services provided through the application context
4. **Decorator Pattern**: For augmenting operators with additional behavior (e.g., tracing)
5. **Strategy Pattern**: Pluggable implementations for core functionality

## Type System

Ember employs a comprehensive type system that enhances code safety and developer experience:

### Core Components

1. **EmberModel**: The foundational model class
   - Extends Pydantic's BaseModel for validation
   - Implements serialization protocols
   - Supports dictionary-like access for compatibility

2. **Protocols**:
   - **EmberTyped**: Interface for type introspection
   - **EmberSerializable**: Interface for data conversion
   - **ConfigManager**: Interface for configuration management
   - **XCSGraph/XCSNode/XCSPlan**: Interfaces for execution components

3. **Generic Types**:
   - **ModelRegistry[ProviderT, ModelT]**: Type-safe model registration
   - **Operator[InputT, OutputT]**: Type-safe operator definitions

4. **Runtime Type Checking**:
   - **validate_type**: Validation against expected types
   - **type_check**: Comprehensive type validation

### Benefits

- **Early Error Detection**: Type errors caught at development time
- **Self-Documenting Code**: Types clearly communicate expectations
- **IDE Support**: Enhanced autocompletion and navigation
- **Refactoring Safety**: Type changes reveal impacted code

### Usage Example

```python
from ember.core.types import EmberModel, InputT, OutputT
from ember.core.registry.operator.base import Operator

class InputData(EmberModel):
    prompt: str
    temperature: float = 0.7

class OutputData(EmberModel):
    response: str
    tokens_used: int

class SimpleOperator(Operator[InputData, OutputData]):
    def forward(self, inputs: InputData) -> OutputData:
        # Type flexibility: Either format below works with the type system
        # Option 1: Return as proper typed model (preferred for IDE support)
        # return OutputData(response="Hello", tokens_used=5)
        
        # Option 2: Return as dictionary - automatically converted to OutputData
        return {"response": "Hello", "tokens_used": 5}
            
# Both styles of input are supported too
operator = SimpleOperator()
# Using typed model
result1 = operator(InputData(prompt="Hi", temperature=0.7))
# Using dictionary (automatically converted to InputData)
result2 = operator({"prompt": "Hi", "temperature": 0.3})
```

## Execution Flow

When executing a typical operation through Ember, the flow is:

1. **Operator Creation**: Instantiate operators with their required configuration
2. **Graph Creation** (optional): Define the execution graph with operators as nodes
3. **Input Preparation**: Prepare the input data for the operators
4. **Execution**:
   - **Eager Mode**: Directly invoke `operator(inputs=...)` for immediate execution
   - **Graph Mode**: Compile the graph into an execution plan and run through a scheduler
5. **Result Processing**: Process and return the execution results

## Extension Points

Ember is designed for extensibility at multiple levels:

1. **New Model Providers**: Implement `BaseProviderModel` and register via discovery
2. **Custom Operators**: Extend the `Operator` base class and implement forward()
3. **Specialized Signatures**: Create custom signatures for typed I/O
4. **Custom Schedulers**: Implement specialized execution strategies
5. **Evaluation Metrics**: Add new metrics to the evaluation system

## Performance Considerations

Ember balances ease of use with performance:

- **Lazy Loading**: Models are instantiated only when needed
- **Parallel Execution**: Graph-based execution can parallelize independent operations
- **Caching**: Discovery results and configurations are cached
- **Thread Safety**: Core components are designed for concurrent access

## Scalability

For large-scale deployments, consider:

1. **Model Caching**: Use the caching mechanisms in ModelService
2. **Rate Limiting**: Configure appropriate rate limits for each provider
3. **Cost Management**: Leverage the UsageService to monitor and control costs
4. **Workload Distribution**: For very large workloads, consider distributing across multiple processes/machines
5. **Memory Management**: Be aware of in-memory data size, especially for large datasets