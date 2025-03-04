# Ember Quickstart Guide

This guide will help you quickly get started with Ember, the compositional framework for building and orchestrating Compound AI Systems.

## Installation

### Option 1: Minimal Installation (OpenAI Only)

```bash
# Install via pip with OpenAI support only
pip install "ember-ai[minimal]"
```

### Option 2: Full Installation (All Features)

```bash
# Install with all providers and features
pip install "ember-ai[all]"
```

### Option 3: Custom Installation

```bash
# Install with specific providers
pip install "ember-ai[openai,anthropic]"

# Install with data processing capabilities
pip install "ember-ai[openai,data]"

# Install for developers
pip install "ember-ai[dev]"
```

## Setting Up API Keys

Set your API keys as environment variables:

```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Windows PowerShell
$env:OPENAI_API_KEY="your-openai-key"
$env:ANTHROPIC_API_KEY="your-anthropic-key"
```

## Basic Usage

Here's how to get started with Ember in just a few lines of code:

```python
# Import the package
import ember

# Initialize and get the model service
service = ember.init()

# Make a simple model call
response = service("openai:gpt-4o", "What is the capital of France?")
print(response.data)
```

## Building Your First Compound AI System

Let's create a simple example that uses multiple models with parallelization:

```python
from typing import ClassVar
from ember.xcs.tracer import jit
from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.core import non

# Define structured input/output models
class QueryInput(EmberModel):
    query: str
    
class QueryOutput(EmberModel):
    answer: str
    confidence: float

# Define the specification
class QuerySpecification(Specification):
    input_model = QueryInput
    output_model = QueryOutput

# Create a compound system using the @jit decorator for optimization
@jit
class ParallelQuerySystem(Operator[QueryInput, QueryOutput]):
    # Class-level specification declaration
    specification: ClassVar[Specification] = QuerySpecification()
    
    # Class-level field declarations
    ensemble: non.UniformEnsemble
    aggregator: non.MostCommon
    
    def __init__(self):
        # Initialize fields
        self.ensemble = non.UniformEnsemble(
            num_units=3,  # Use 3 models in parallel
            model_name="openai:gpt-4o-mini",
            temperature=0.4
        )
        
        self.aggregator = non.MostCommon()
    
    def forward(self, *, inputs: QueryInput) -> QueryOutput:
        # Get responses from multiple models (automatically parallelized)
        ensemble_result = self.ensemble(inputs={"query": inputs.query})
        
        # Aggregate the results
        aggregated = self.aggregator(inputs={
            "query": inputs.query,
            "responses": ensemble_result["responses"]
        })
        
        # Return structured output
        return QueryOutput(
            answer=aggregated["final_answer"],
            confidence=aggregated.get("confidence", 0.0)
        )

# Create and use the system
system = ParallelQuerySystem()
result = system(inputs={"query": "What is the speed of light?"})

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2f}")
```

## Next Steps

- Explore the [Model Registry](docs/quickstart/model_registry.md) for using different LLM providers
- Learn about [Operators](docs/quickstart/operators.md) for building reusable components
- Check out [Networks of Networks](docs/quickstart/non.md) for complex AI systems
- Set up [Data Processing](docs/quickstart/data.md) for dataset handling

For a full walkthrough of Ember's capabilities, see the [Examples Directory](src/ember/examples).

## Getting Help

- Documentation: [https://docs.pyember.org](https://docs.pyember.org)
- GitHub Issues: [https://github.com/pyember/ember/issues](https://github.com/pyember/ember/issues)
- Discord Community: [https://discord.gg/ember-ai](https://discord.gg/ember-ai)