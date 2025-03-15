# Ember: Compositional Framework for Compound AI Systems

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Ember in a Nutshell

Ember is what PyTorch and JAX were for neural networks, but for the era of compound AI systems - a compositional framework with both eager execution and graph optimization capabilities. It enables you to compose complex "Networks of Networks" (NONs) with automatic parallelization and optimization.

## Quick Example: Multi-Model Ensemble with Synthesis

```python
from typing import ClassVar
from ember.api.operator import Operator
from ember.api.xcs import jit
from ember.api import non
from ember.api.models import EmberModel

# Define structured I/O types, in a TypedDict syntax
class QueryInput(EmberModel):
    query: str
    
class ReasonedOutput(EmberModel):
    answer: str
    confidence: float

@jit  # Automatically optimize execution
class EnsembleReasoner(Operator[QueryInput, ReasonedOutput]):
    """A multi-model ensemble with synthesis for robust reasoning."""
    ensemble: non.UniformEnsemble
    judge: non.JudgeSynthesis
    
    def __init__(self):
        # Create components for the reasoning pipeline
        self.ensemble = non.UniformEnsemble(
            num_units=3,
            model_name="openai:gpt-4o",
            temperature=0.7
        )
        
        self.judge = non.JudgeSynthesis(
            model_name="anthropic:claude-3-sonnet",
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

1. **Composable Operators with rigorous specification**: Build richer compound AI systems (CAIS) from reusable components
2. **Automatic Parallelization**: Across a full CAIS DAG Independent operations can execute concurrently
3. **_Bonus (supplemental)_** **Multi-Provider Support** -- Unified API across OpenAI, Anthropic, Gemini, and more

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

Compose more complex CAIS architectures using pre-built components:

```python
from ember.api import non

# Create an ensemble of 5 model instances that run in parallel
ensemble = non.UniformEnsemble(
    num_units=5, 
    model_name="openai:gpt-4o-mini",
    temperature=0.7
)

# Create a verifier to check outputs
verifier = non.Verifier(
    model_name="anthropic:claude-3-sonnet",
    verification_criteria=["factual_accuracy", "coherence"]
)

# Create an aggregator that selects the most common answer
aggregator = non.MostCommon()

# These components can be composed in custom operators
```

## Graph Optimization

Ember's XCS system provides XLA-inspired tracing and optimization:

```python
from ember.api.xcs import jit, execution_options

@jit  # Automatically optimize execution
class ComplexPipeline(Operator):
    # ... operator implementation ...

# Configure execution parameters
with execution_options(max_workers=8):
    result = pipeline(query="Complex question...") 
```

## Data Handling & Evaluation

Load, transform, and evaluate with standard datasets:

```python
from ember.api.data import DataLoader, EvaluationPipeline, Evaluator

# Load a dataset
loader = DataLoader.from_registry("mmlu")
dataset = loader.load(subset="physics", split="test")

# Create an evaluation pipeline
eval_pipeline = EvaluationPipeline([
    Evaluator.from_registry("accuracy"),
    Evaluator.from_registry("response_quality")
])

# Evaluate a model or operator
results = eval_pipeline.evaluate(my_model, dataset)
print(f"Accuracy: {results['accuracy']:.2f}")
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