# Ember Architecture

This document describes Ember's architecture, core components, and design principles. It serves as both a high-level overview for users and a detailed guide for contributors.

## Design Philosophy

Ember is built on these foundational principles:

1. **Composability First**: The ability to combine, chain, and nest components is central to Ember's design
2. **Type Safety**: Comprehensive type annotations ensure robustness and IDE support
3. **Testability**: Components are designed for easy isolation and testing
4. **Performance**: Parallel execution is built-in at the framework's core
5. **Extensibility**: Registry-based design makes it simple to add new components
6. **Developer Experience**: APIs follow familiar patterns from PyTorch/JAX

## System Architecture

Ember's architecture follows a layered design with clear separations of concern:

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                    APPLICATION LAYER                                       │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │  NON Patterns           │  │  Auto Graph Builder     │  │  Enhanced JIT            │   │
│  │                         │  │                         │  │                         │   │
│  │  • UniformEnsemble      │  │  • Autograph            │  │  • Function Tracing     │   │
│  │  • JudgeSynthesis       │  │  • Graph Construction   │  │  • Optimized Execution  │   │
│  │  • Verifier             │  │  • Dependency Tracking  │  │  • Parallel Dispatch    │   │
│  │  • VariedEnsemble       │  │  • Visualization        │  │  • Runtime Fusion       │   │
│  └─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘   │
│                                                                                           │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                      CORE COMPONENT LAYER                                 │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │  Model Registry         │  │  Operator System        │  │  Prompt Specifications      │   │
│  │                         │  │                         │  │                         │   │
│  │  • ModelInfo            │  │  • Base Operator        │  │  • Template Rendering   │   │
│  │  • ModelService         │  │  • Operator Registry    │  │  • Input Validation     │   │
│  │  • UsageService         │  │  • Core Operators       │  │  • Output Validation    │   │
│  │  • Provider Adapters    │  │  • Custom Operators     │  │  • Schema Generation    │   │
│  └─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘   │
│                                                                                           │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │  Data Processing        │  │  Evaluation Tools       │  │  Application Context    │   │
│  │                         │  │                         │  │                         │   │
│  │  • Dataset Loaders      │  │  • Evaluators           │  │  • Config Management    │   │
│  │  • Transformers         │  │  • Metrics              │  │  • Dependency Injection │   │
│  │  • Samplers             │  │  • Result Analysis      │  │  • Service Registry     │   │
│  │  • Benchmark Registry   │  │  • Visualization        │  │  • Logging              │   │
│  └─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘   │
│                                                                                           │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                  EXECUTION ENGINE (XCS)                                   │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │  Graph Definition       │  │  Tracer System          │  │  Execution Engine       │   │
│  │                         │  │                         │  │                         │   │
│  │  • XCSGraph             │  │  • Function Tracing     │  │  • Schedulers           │   │
│  │  • XCSNode              │  │  • Execution Recording  │  │  • Execution Plan       │   │
│  │  • Edge Management      │  │  • JIT Compilation      │  │  • Parallel Dispatch    │   │
│  │  • Graph Visualization  │  │  • Graph Optimization   │  │  • Error Handling       │   │
│  └─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘   │
│                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### 1. Execution Engine (XCS)

The foundational layer providing computation graph definition and execution:

* **Graph Definition**: Defines the structure of computation
* **Tracer System**: Records execution and enables optimization
* **Execution Engine**: Manages the actual running of operations with parallelization

### 2. Core Component Layer

The building blocks of Ember's functionality:

* **Model Registry**: Management of LLM providers and models
* **Operator System**: Core computational units and their registry
* **Prompt Specifications**: Type-safe template rendering and validation
* **Data Processing**: Dataset handling, transformation, and sampling
* **Evaluation Tools**: Benchmarking and performance analysis
* **Application Context**: Configuration and dependency management

### 3. Application Layer

High-level abstractions built on the core components:

* **NON Patterns**: Ready-to-use Networks of Networks patterns
* **Auto Graph Builder**: Automatic graph construction from code
* **Enhanced JIT**: Just-in-time compilation for optimized execution

## Component Details

### Application Context

The `EmberAppContext` serves as the central dependency injection container:

```python
from ember.core.app_context import get_app_context

# Access global context
context = get_app_context()

# Access services
model_service = context.model_service
usage_service = context.usage_service

# Access configuration
config = context.config_manager.get_config("model_registry")
```

Key responsibilities:
- Initialization and configuration of system components
- Service registration and dependency injection
- Configuration management
- Logging setup

### Model Registry System

The Model Registry manages connections to LLM providers:

```python
from ember.core.registry.model import ModelRegistry, ModelInfo
from ember.core.registry.model.base.services import ModelService

# Register a model
registry = ModelRegistry()
registry.register_model(ModelInfo(id="openai:gpt-4o", ...))

# Create a service for model access
service = ModelService(registry=registry)
response = service.invoke_model("openai:gpt-4o", "Hello world")
```

#### Model Registry Component Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                           Model Registry System                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │  ModelRegistry  │◄────►│  ModelFactory   │─────►│ Provider Models │ │
│  └────────┬────────┘      └─────────────────┘      └─────────────────┘ │
│           │                                                             │
│           │               ┌─────────────────┐      ┌─────────────────┐ │
│           └──────────────►│  ModelService   │◄────►│  UsageService   │ │
│                           └─────────────────┘      └─────────────────┘ │
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │ OpenAI Provider │      │Anthropic Provider│      │ Other Providers │ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Key components:
- `ModelRegistry`: Central repository for model metadata
- `ModelService`: High-level API for model invocation
- `UsageService`: Tracks token usage and cost
- Provider implementations: OpenAI, Anthropic, Google, IBM, etc.

### Operator System

Operators are the fundamental computational units in Ember:

```python
from ember.core.registry.operator.base import Operator
from pydantic import BaseModel

class SummarizerInput(BaseModel):
    text: str
    max_words: int = 100

class SummarizerOutput(BaseModel):
    summary: str
    word_count: int

class SummarizerOperator(Operator[SummarizerInput, SummarizerOutput]):
    def forward(self, *, inputs: SummarizerInput) -> SummarizerOutput:
        # Implementation
        ...
```

#### Operator System Component Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                           Operator System                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │OperatorRegistry │◄────►│OperatorFactory  │─────►│Operator Instance│ │
│  └─────────────────┘      └─────────────────┘      └────────┬────────┘ │
│                                                              │         │
│  ┌─────────────────┐      ┌─────────────────┐      ┌────────▼────────┐ │
│  │  Base Operator  │◄─────┤ Prompt Specification │◄────►│   forward()    │ │
│  └────────┬────────┘      └─────────────────┘      └─────────────────┘ │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │ Core Operators  │      │ Custom Operators │      │  NON Operators  │ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Key components:
- `Operator`: Base class for all operators
- `Specification`: Type definitions for operator I/O
- Core operators: Ensemble, Judge, Verifier, etc.
- Operator registry for discovery

### Prompt Specification System

Specifications define the contract between inputs and outputs:

```python
from ember.core.registry.prompt_specification import Specification
from pydantic import BaseModel

class QuestionInput(BaseModel):
    question: str
    context: str

class AnswerOutput(BaseModel):
    answer: str
    confidence: float

class QASpecification(Specification):
    input_model = QuestionInput
    structured_output = AnswerOutput
    prompt_template = """
    Answer the question based on the context.
    
    Context: {context}
    Question: {question}
    """
```

#### Prompt Specification Component Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                          Prompt Specification System                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │    Specification    │──────┤   Input Model   │      │  Output Model   │ │
│  └────────┬────────┘      └─────────────────┘      └─────────────────┘ │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │Prompt Template  │─────►│Template Renderer │─────►│Input Validation │ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │Schema Generation│◄─────┤Output Validation │◄─────┤  Error Handling │ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Key features:
- Type validation for inputs and outputs
- Template rendering with validation
- Automatic placeholder checking
- Support for structured data extraction

### Execution Engine (XCS)

The XCS (eXecution Control System) handles graph-based execution:

```python
from ember.xcs.graph import XCSGraph
from ember.xcs.engine import execute_graph

# Create execution graph
graph = XCSGraph()
graph.add_node(operator=ensemble, node_id="ensemble")
graph.add_node(operator=judge, node_id="judge") 
graph.add_edge(from_id="ensemble", to_id="judge")

# Execute with parallelization
result = execute_graph(
    graph=graph,
    global_input={"query": "What is quantum computing?"},
    max_workers=3
)
```

#### XCS Engine Component Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                        Execution Engine (XCS)                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │    XCSGraph     │─────►│    XCSNode      │◄─────┤      Edge       │ │
│  └────────┬────────┘      └────────┬────────┘      └─────────────────┘ │
│           │                        │                                    │
│           ▼                        ▼                                    │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │  Graph Compiler │─────►│ Execution Plan  │─────►│    Scheduler    │ │
│  └─────────────────┘      └────────┬────────┘      └────────┬────────┘ │
│                                    │                         │         │
│                                    ▼                         ▼         │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │  Input Mapping  │◄─────┤ Parallel Worker │◄─────┤ Output Collection│ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Key components:
- `XCSGraph`: Directed acyclic graph representation
- `ExecutionPlan`: Compiled execution plan
- `Scheduler`: Controls execution strategy
- `Tracer`: Records execution for debugging

### Enhanced JIT System

The JIT system enables automatic graph construction and optimization:

```python
from ember.xcs.tracer import jit

@jit
def complex_pipeline(query: str):
    # Operations are automatically traced and optimized
    ensemble = UniformEnsemble(num_units=3, model_name="openai:gpt-4o")
    responses = ensemble({"query": query})
    
    judge = JudgeSynthesis(model_name="anthropic:claude-3-sonnet")
    result = judge({"query": query, "responses": responses.responses})
    
    return result.final_answer
```

#### JIT System Component Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                          Enhanced JIT System                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │   JIT Decorator │─────►│Function Tracing  │─────►│ Graph Building  │ │
│  └────────┬────────┘      └────────┬────────┘      └────────┬────────┘ │
│           │                        │                         │         │
│           ▼                        ▼                         ▼         │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │   Call Cache    │◄─────┤ Graph Optimizer │◄─────┤Dependency Tracker│ │
│  └─────────────────┘      └────────┬────────┘      └─────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │Parallel Dispatch│◄─────┤Execution Context│─────►│Result Collection │ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Key features:
- Automatic graph construction
- Function tracing
- Parallelization of independent operations
- Caching and optimization

### Data Processing System

The data module provides tools for dataset management:

```python
from ember.core.utils.data.service import DataService
from ember.core.utils.data.base.samplers import RandomSampler

# Load a benchmark dataset
data_service = DataService()
mmlu_data = data_service.load_dataset(
    dataset_name="mmlu",
    subset="high_school_mathematics",
    sampler=RandomSampler(n=100)
)
```

#### Data Processing Component Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                         Data Processing System                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │  DataService    │─────►│ Dataset Registry │─────►│ Dataset Loaders │ │
│  └────────┬────────┘      └─────────────────┘      └────────┬────────┘ │
│           │                                                  │         │
│           ▼                                                  ▼         │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │ Dataset Cache   │◄─────┤  Dataset Item   │◄─────┤  External API   │ │
│  └─────────────────┘      └────────┬────────┘      └─────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │ Data Transformer│◄─────┤ Data Sampler    │─────►│ Data Validator  │ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Key components:
- `DataService`: Central access point for datasets
- Dataset loaders for popular benchmarks
- Transformers for data preprocessing
- Samplers for dataset subsampling

### Evaluation System

The evaluation system measures model performance:

```python
from ember.core.utils.eval.pipeline import EvaluationPipeline
from ember.core.utils.eval.evaluators import MultipleChoiceEvaluator

# Create evaluation pipeline
eval_pipeline = EvaluationPipeline(
    dataset=test_data,
    evaluators=[MultipleChoiceEvaluator()],
    model=model
)

# Run evaluation
results = eval_pipeline.evaluate()
print(f"Accuracy: {results.metrics['accuracy']:.2f}")
```

#### Evaluation Component Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                          Evaluation System                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │  Eval Pipeline  │─────►│  Eval Registry  │─────►│    Evaluator    │ │
│  └────────┬────────┘      └─────────────────┘      └────────┬────────┘ │
│           │                                                  │         │
│           ▼                                                  ▼         │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │ Data Provider   │─────►│ Model Runner    │─────►│ Result Collector│ │
│  └─────────────────┘      └─────────────────┘      └────────┬────────┘ │
│                                                              │         │
│                                                              ▼         │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │ Metric Calculator│◄────┤ Result Analyzer │◄─────┤ Report Generator│ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Key components:
- `EvaluationPipeline`: Orchestrates evaluation
- Task-specific evaluators
- Metrics collection
- Result reporting

## Full System Dependency Flow

The diagram below illustrates the complete dependency flow between major components:

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Configuration Layer                                      │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐ │
│  │    Config Files         │────►│    Config Manager       │────►│   Environment       │ │
│  │    (.yaml, .env)        │     │                         │     │   Variables         │ │
│  └─────────────────────────┘     └───────────┬─────────────┘     └─────────────────────┘ │
│                                              │                                            │
│                                              ▼                                            │
│                                 ┌─────────────────────────┐                              │
│                                 │    EmberAppContext      │                              │
│                                 └───────────┬─────────────┘                              │
│                                             │                                             │
└─────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                    Service Layer                                          │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐ │
│  │    Model Registry       │◄───►│     Model Service       │◄───►│   Usage Service     │ │
│  └───────────┬─────────────┘     └───────────┬─────────────┘     └─────────────────────┘ │
│              │                               │                                            │
│              ▼                               ▼                                            │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐ │
│  │   Provider Models       │◄───►│    Operator Registry    │◄───►│  Data Service       │ │
│  └─────────────────────────┘     └───────────┬─────────────┘     └───────────┬─────────┘ │
│                                              │                               │           │
└─────────────────────────────────────────────┼───────────────────────────────┼─────────────┘
                                              │                               │
                                              ▼                               ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                    Component Layer                                        │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐ │
│  │    Base Operators       │◄───►│    Prompt Specifications    │◄───►│   Dataset Loaders   │ │
│  └───────────┬─────────────┘     └─────────────────────────┘     └─────────────────────┘ │
│              │                                                                            │
│              ▼                                                                            │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐ │
│  │   Core Operators        │◄───►│     NON Patterns        │◄───►│   Evaluators        │ │
│  └───────────┬─────────────┘     └───────────┬─────────────┘     └─────────────────────┘ │
│              │                               │                                            │
└─────────────┼───────────────────────────────┼────────────────────────────────────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Execution Engine Layer                                   │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐ │
│  │       XCSGraph          │◄───►│     Graph Compiler      │◄───►│   JIT Compiler      │ │
│  └───────────┬─────────────┘     └───────────┬─────────────┘     └───────────┬─────────┘ │
│              │                               │                               │           │
│              ▼                               ▼                               ▼           │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐ │
│  │    Execution Plan       │◄───►│      Scheduler          │◄───►│  Parallel Executor  │ │
│  └─────────────────────────┘     └─────────────────────────┘     └─────────────────────┘ │
│                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## Code Organization

The code is organized into the following package structure:

| Package | Purpose |
|---------|---------|
| `ember.core` | Core framework classes and utilities |
| `ember.core.app_context` | Application context and DI container |
| `ember.core.configs` | Configuration management |
| `ember.core.types` | Type system, protocols, and validation |
| `ember.core.registry.model` | Model registry and provider implementations |
| `ember.core.registry.operator` | Operator system |
| `ember.core.registry.prompt_specification` | Prompt specification system |
| `ember.core.utils` | Utility functions and helpers |
| `ember.core.utils.data` | Data processing and datasets |
| `ember.core.utils.eval` | Evaluation and metrics |
| `ember.core.non` | High-level NON patterns |
| `ember.xcs` | Execution engine |
| `ember.xcs.graph` | Graph definition and manipulation |
| `ember.xcs.engine` | Execution scheduling |
| `ember.xcs.tracer` | Tracing and JIT compilation |

## Design Patterns

Ember employs several design patterns that are consistent throughout the codebase:

### 1. Registry Pattern

```python
# Registry implementation
class ModelRegistry:
    def __init__(self):
        self._models = {}
    
    def register_model(self, model_info: ModelInfo) -> None:
        self._models[model_info.id] = model_info
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        return self._models[model_id]
        
# Usage
registry = ModelRegistry()
registry.register_model(ModelInfo(id="model1", ...))
model = registry.get_model_info("model1")
```

### 2. Factory Pattern

```python
class ModelFactory:
    def create_model(self, model_info: ModelInfo) -> BaseProviderModel:
        provider_name = model_info.provider["name"]
        if provider_name == "OpenAI":
            return OpenAIModel(model_info)
        elif provider_name == "Anthropic":
            return AnthropicModel(model_info)
        # etc.
```

### 3. Dependency Injection

```python
class ModelService:
    def __init__(self, registry: ModelRegistry, usage_service: Optional[UsageService] = None):
        self.registry = registry
        self.usage_service = usage_service
```

### 4. Composition Pattern

```python
class EnsembleOperator(Operator[EnsembleInput, EnsembleOutput]):
    def __init__(self, lm_modules: List[LMModule]):
        self.lm_modules = lm_modules
    
    def forward(self, *, inputs: EnsembleInput) -> EnsembleOutput:
        # Use contained modules
        responses = [lm(inputs.query) for lm in self.lm_modules]
        return EnsembleOutput(responses=responses)
```

### 5. Strategy Pattern

```python
class Scheduler(Protocol):
    def run_plan(self, plan: ExecutionPlan, global_input: Dict, graph: XCSGraph) -> Any:
        ...

class SerialScheduler:
    def run_plan(self, plan: ExecutionPlan, global_input: Dict, graph: XCSGraph) -> Any:
        # Serial execution implementation

class ParallelScheduler:
    def run_plan(self, plan: ExecutionPlan, global_input: Dict, graph: XCSGraph) -> Any:
        # Parallel execution implementation
```

## Performance Considerations

Ember balances ease of use with high performance:

### Parallelization Strategy

1. **Graph-Based Parallelism**: 
   - The XCS engine automatically identifies independent operations
   - Executes them concurrently using thread pools
   - Configurable max_workers parameter

2. **Operator-Level Concurrency**: 
   - Operators can implement their own internal parallelism
   - Example: EnsembleOperator runs multiple models concurrently

3. **Efficient Resource Usage**:
   - Smart thread pooling to avoid over-subscription
   - Rate limiting to respect API constraints

### Memory Management

1. **Lazy Instantiation**:
   - Models are instantiated only when needed
   - Heavy resources are loaded on demand

2. **Caching Strategy**:
   - Configuration is cached after initial load
   - Discovery results are cached
   - Model instances are reused

### Optimization Techniques

1. **JIT Compilation**:
   - Traces function execution to build optimized graphs
   - Identifies parallelizable operations
   - Minimizes redundant computations

2. **Efficient Data Transfer**:
   - Minimizes copying of large data between operators
   - Uses references when possible

## Deployment Considerations

When deploying Ember in production, consider these best practices:

### 1. Configuration Management

- Store API keys securely in environment variables
- Use separate configurations for development/production
- Override defaults with environment-specific settings

```
# Development configuration
config/
  base.yaml        # Base configuration for all environments
  development.yaml # Development-specific overrides
  production.yaml  # Production-specific overrides
```

### 2. Resource Planning

- Set appropriate thread pool sizes (max_workers)
- Monitor token usage with UsageService
- Implement rate limiting strategies
- Set up cost budgets and alerts

### 3. Error Handling

- Implement proper error handling at the application level
- Set up exponential backoff for API rate limits
- Use the retry utilities for transient errors
- Log errors comprehensively

### 4. Monitoring and Observability

- Set up proper logging with appropriate log levels
- Monitor token and request metrics
- Track performance of individual operators
- Set alerts for abnormal behavior

### 5. Scaling Strategies

For high-throughput applications:
- Distribute workloads across multiple processes or machines
- Use horizontal scaling for independent operations
- Consider specialized execution engines for very large workloads
- Use caching for frequently used models or operations

## Request Flow Diagram

The following diagram illustrates the flow of a typical request through the Ember system:

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│              │      │              │      │              │      │              │
│   User API   │─────►│ ModelService │─────►│ ModelRegistry│─────►│ ModelFactory │
│  Request     │      │              │      │              │      │              │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │                     │
       │                     │                     │                     ▼
       │                     │                     │              ┌──────────────┐
       │                     │                     │              │              │
       │                     │                     └─────────────►│ Provider     │
       │                     │                                    │ Implementation│
       │                     │                                    └──────┬───────┘
       │                     │                                           │
       │                     │                                           ▼
       │                     │                                    ┌──────────────┐
       │                     │                                    │              │
       │                     └───────────────────────────────────►│ UsageService │
       │                                                          │              │
       │                                                          └──────┬───────┘
       │                                                                 │
       ▼                                                                 ▼
┌──────────────┐                                                  ┌──────────────┐
│              │                                                  │              │
│  User API    │◄─────────────────────────────────────────────────┤   Response   │
│  Response    │                                                  │              │
└──────────────┘                                                  └──────────────┘
```

## Architecture Evolution

The Ember architecture continues to evolve along these paths:

1. **Distributed Execution**: Support for distributed execution across multiple machines
2. **Enhanced Caching**: Improved caching for models and intermediate results
3. **Custom Hardware Support**: Optimizations for specialized hardware (GPUs, TPUs)
4. **Plugin System**: More comprehensive plugin interfaces for extensions
5. **Advanced Graph Optimizations**: Additional graph transformations and optimizations

## Additional Resources

For more detailed information, consult these resources:

- [Model Registry Documentation](docs/quickstart/model_registry.md)
- [Operator System Documentation](docs/quickstart/operators.md)
- [XCS Execution Engine Documentation](docs/advanced/xcs_graphs.md)
- [Enhanced JIT Documentation](docs/advanced/enhanced_jit.md)
- [Example Applications](src/ember/examples)