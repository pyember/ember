# Ember Quickstart Guide

This guide will help you quickly get started with Ember, the compositional framework for building and orchestrating Compound AI Systems.

## Installation

```bash
# Clone the repository
git clone https://github.com/pyember/ember.git
cd ember

# Install using Poetry
poetry install

# For development dependencies
poetry install --with dev
```

## Setting Up API Keys and Configuration

### Option 1: Environment Variables

Set your API keys as environment variables:

```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Windows PowerShell
$env:OPENAI_API_KEY="your-openai-key"
$env:ANTHROPIC_API_KEY="your-anthropic-key"
```

### Option 2: Configuration File

Create a configuration file at `~/.ember/config.yaml` or in your project directory:

```yaml
model_registry:
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}  # Will use environment variable
    anthropic:
      api_key: "your-anthropic-key"  # Direct value
```

See [Configuration Quickstart](docs/quickstart/configuration.md) for more options.

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
from ember.api.xcs import jit
from ember.api.operator import Operator, Specification
from ember.api.models import EmberModel
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
    structured_output = QueryOutput

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
- Configure your application with [Configuration](docs/quickstart/configuration.md)
- Use [Simplified Imports](SIMPLIFIED_IMPORTS.md) for cleaner code

To see Ember in action, explore these key examples:
- [Minimal Example](src/ember/examples/minimal_example.py) - Get started with basic usage
- [Model API Example](src/ember/examples/model_api_example.py) - Learn the models API
- [Ensemble Operator Example](src/ember/examples/diverse_ensemble_operator_example.py) - Build parallel model ensembles
- [API Operators Example](src/ember/examples/api_operators_example.py) - Use streamlined imports
- [Enhanced JIT Example](src/ember/examples/enhanced_jit_example.py) - Optimize execution with JIT

For a full walkthrough of Ember's capabilities, see the [Examples Directory](src/ember/examples).

## Getting Help

- GitHub Issues: [https://github.com/pyember/ember/issues](https://github.com/pyember/ember/issues)
- Documentation: See the documentation files in the `docs/` directory
- Examples: Explore the examples in `src/ember/examples/`