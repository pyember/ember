# Ember: Compositional Framework for Compound AI Systems

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Ember in a Nutshell

Aspirationally, Ember is to Networks of Networks (NONs) Compound AI Systems development what PyTorch and XLA are to Neural Networks (NN) development. It's a compositional framework with both eager execution affordances and graph execution optimization capabilities. It enables users to compose complex NONs, and supports automatic parallelization and optimization of these.

## Simple Example: `horizontal` inference-time scaling with best-of-N

```python
from typing import ClassVar
from ember.api.operator import Operator, Specification
from ember.api.xcs import jit
from ember.api import non
from ember.api.models import EmberModel

# Define structured I/O types, in a TypedDict syntax
class QueryInput(EmberModel):
    query: str
    
class ReasonedOutput(EmberModel):
    answer: str
    confidence: float

class ReasonerSpecification(Specification):
    input_model = QueryInput
    structured_output = ReasonedOutput

@jit  # Automatically optimize execution with JIT compilation
class EnsembleReasoner(Operator[QueryInput, ReasonedOutput]):
    """A multi-model ensemble with synthesis for robust reasoning."""
    # Input/output specification
    specification: Specification = ReasonerSpecification()

    # Sub-operators
    ensemble: non.UniformEnsemble
    judge: non.JudgeSynthesis
    
    def __init__(self, width: int = 3):
        # Create components for the reasoning pipeline
        self.ensemble = non.UniformEnsemble(
            num_units=width,
            model_name="openai:gpt-4o",
            temperature=0.7
        )
        
        self.judge = non.JudgeSynthesis(
            model_name="anthropic:claude-3.5-sonnet",
            temperature=0.2
        )
    
    def forward(self, *, inputs: QueryInput) -> ReasonedOutput:
        # Get multiple reasoning paths (executed in parallel)
        ensemble_result = self.ensemble(query=inputs.query)
        
        # Synthesize a final response 
        synthesis = self.judge(
            query=inputs.query,
            responses=ensemble_result["responses"]
        )
        
        # Return structured output
        return ReasonedOutput(
            answer=synthesis["final_answer"],
            confidence=float(synthesis.get("confidence", 0.0))
        )

# Use it like any Python function
reasoner = EnsembleReasoner()
result = reasoner(query="What causes the northern lights?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2f}")
```

## Core Elements

1. **Composable Operators with Rigorous Specification**: Build reliable compound AI systems from type-safe, reusable components with validated inputs and outputs
2. **Automatic Parallelization**: Independent operations are automatically executed concurrently across a full computational graph
3. **XCS Optimization Framework**: Just-in-time tracing and execution optimization inspired by JAX/XLA 
4. **Multi-Provider Support**: Unified API across OpenAI, Anthropic, Claude, Gemini, and more with standardized usage tracking

## Installation

```bash
# Install using pip
pip install ember-ai

# Or from source with Poetry
git clone https://github.com/pyember/ember.git
cd ember
poetry install
```

## Model Registry & Provider Integration

Access models from any provider through a unified interface:

```python
from ember import initialize_ember
from ember.api.models import ModelEnum

# Initialize with multiple providers
service = initialize_ember(usage_tracking=True)

# Access models from different providers with the same API
response = service(ModelEnum.OPENAI_GPT4O, "What is quantum computing?")
print(response.data)

# Track usage across providers
usage = service.usage_service.get_total_usage()
print(f"Total cost: ${usage.cost:.4f}")
```

## NON Patterns & Ensembling

Build compound AI system architectures using the Network of Networks (NON) pattern with pre-built components:

```python
from ember.api import non

# 1. Create an ensemble of 5 model instances that run in parallel
# This improves statistical robustness through multiple independent samples
ensemble = non.UniformEnsemble(
    num_units=5, 
    model_name="openai:gpt-4o-mini",
    temperature=0.7
)

# 2. Create a judge to synthesize the ensemble responses
# This combines multiple perspectives into a coherent, reasoned answer
judge = non.JudgeSynthesis(
    model_name="anthropic:claude-3-sonnet",
    temperature=0.2
)

# 3. Create a verifier to independently check the final output
# This adds a quality control layer for factual accuracy and coherence
verifier = non.Verifier(
    model_name="anthropic:claude-3-haiku",
    temperature=0.0
)

# 4. Chain components together in a sequential pipeline
# Ember automatically optimizes the execution graph
pipeline = non.Sequential(operators=[ensemble, judge, verifier])

# Execute the entire pipeline with a single call
result = pipeline(query="What causes tsunamis?")
```

## Graph Optimization & Execution

Ember's XCS system provides JAX/XLA-inspired tracing, transformation, and automatic parallelization:

```python
from ember.api.xcs import jit, structural_jit, execution_options, vmap

# Basic JIT compilation for simple optimization
@jit
class SimplePipeline(Operator):
    # ... operator implementation ...

# Advanced structural JIT with parallel execution strategy
@structural_jit(execution_strategy="parallel")
class ComplexPipeline(Operator):
    def __init__(self):
        self.op1 = SubOperator1()
        self.op2 = SubOperator2()
        self.op3 = SubOperator3()
    
    def forward(self, *, inputs):
        # These operations will be automatically parallelized
        # when the execution graph is built
        result1 = self.op1(inputs=inputs)
        result2 = self.op2(inputs=inputs)
        combined = self.op3(inputs={"r1": result1, "r2": result2})
        return combined

# Configure execution parameters
with execution_options(max_workers=8):
    result = pipeline(query="Complex question...") 

# Vectorized mapping for batch processing
@vmap
def process_batch(inputs, model):
    return model(inputs)
```

## Data Handling & Evaluation

Ember provides a comprehensive data processing and evaluation framework with pre-built datasets and metrics:

```python
from ember.api.data import DatasetBuilder, EvaluationPipeline, Evaluator

# Load a dataset with the builder pattern
dataset = (DatasetBuilder()
    .from_registry("mmlu")  # Use a registered dataset
    .subset("physics")      # Select a specific subset
    .split("test")          # Choose the test split
    .sample(100)            # Random sample of 100 items
    .transform(              # Apply transformations
        lambda x: {"query": f"Question: {x['question']}"} 
    )
    .build())

# Create a comprehensive evaluation pipeline
eval_pipeline = EvaluationPipeline([
    # Standard metrics
    Evaluator.from_registry("accuracy"),
    Evaluator.from_registry("response_quality"),
    
    # Custom evaluation metrics
    Evaluator.from_function(
        lambda prediction, reference: {"factual_accuracy": score_factual_content(prediction, reference)}
    )
])

# Evaluate a model or operator
results = eval_pipeline.evaluate(my_model, dataset)
print(f"Accuracy: {results['accuracy']:.2f}")
print(f"Response Quality: {results['response_quality']:.2f}")
print(f"Factual Accuracy: {results['factual_accuracy']:.2f}")
```

## Documentation & Examples

- [Architecture Overview](ARCHITECTURE.md)
- [Quick Start Guide](docs/quickstart/README.md)
- [Model Registry Guide](docs/quickstart/model_registry.md)
- [Operators Guide](docs/quickstart/operators.md)
- [NON Patterns](docs/quickstart/non.md)
- [Data Processing](docs/quickstart/data.md)
- [Configuration](docs/quickstart/configuration.md)
- [Examples Directory](src/ember/examples)

## License

Ember is released under the [Apache 2.0 License](LICENSE).