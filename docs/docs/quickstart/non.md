# Ember NON (Networks of Networks)

This guide introduces Ember's Networks of Networks (NON) module, which provides composable patterns for building robust CAIS workflows.

## Core Operators

The NON module provides reusable operator implementations:

```python
from ember.core.non import (
    UniformEnsemble,    # Multiple identical model instances
    VariedEnsemble,     # Different model configurations  
    JudgeSynthesis,     # Analyze and synthesize multiple responses
    MostCommon,         # Statistical majority voting
    Verifier,           # Validate and correct candidate answers
    Sequential          # Chain multiple operators
)
```

## Basic Usage

### UniformEnsemble

Creates multiple instances of the same model to mitigate non-determinism.

```python
# Create an ensemble with 3 identical instances
ensemble = UniformEnsemble(
    num_units=3,
    model_name="openai:gpt-4o",
    temperature=0.7,
    max_tokens=100
)

# Execute the ensemble
result = ensemble(inputs={"query": "What causes earthquakes?"})
responses = result.responses  # List of responses from each model
```

### JudgeSynthesis

Analyzes multiple responses and synthesizes a better answer.

```python
# Create a judge with a high-quality model
judge = JudgeSynthesis(
    model_name="anthropic:claude-3-sonnet",
    temperature=0.1
)

# Synthesize responses
result = judge(inputs={
    "query": "What causes earthquakes?",
    "responses": ensemble_responses
})

final_answer = result.final_answer
```

### MostCommon

Implements majority voting across multiple responses.

```python
# Create a most-common selector
majority = MostCommon()

# Find the most common answer
result = majority(inputs={
    "query": "What is 2+2?",
    "responses": ["4", "4", "3", "4", "5"]
})

most_common_answer = result.final_answer  # "4"
```

### Verifier

Checks answers for correctness and provides corrections when needed.

```python
# Create a verification operator
verifier = Verifier(
    model_name="openai:gpt-4o",
    temperature=0.2
)

# Verify an answer
result = verifier(inputs={
    "query": "What is the capital of Australia?",
    "candidate_answer": "Sydney is the capital of Australia."
})

verdict = result.verdict           # "incorrect"
explanation = result.explanation   # Explains the error
revised_answer = result.revised_answer  # "Canberra is the capital of Australia."
```

## Building Complex Pipelines

### Ensemble-Judge-Verifier Pipeline

Create a robust multi-step pipeline with JIT optimization:

```python
from ember.core.non import UniformEnsemble, JudgeSynthesis, Verifier, Sequential
from ember.xcs import jit

# Define a JIT-optimized pipeline class
@jit
class RobustQAPipeline(Sequential):
    """Pipeline that combines ensemble, judge, and verification steps."""
    
    ensemble: UniformEnsemble
    judge: JudgeSynthesis
    verifier: Verifier
    
    def __init__(self):
        # Create the ensemble operator
        self.ensemble = UniformEnsemble(
            num_units=3, 
            model_name="openai:gpt-4o",
            temperature=0.7
        )
        
        # Create the judge operator
        self.judge = JudgeSynthesis(
            model_name="anthropic:claude-3-opus",
            temperature=0.1
        )
        
        # Create the verifier operator
        self.verifier = Verifier(
            model_name="openai:gpt-4o",
            temperature=0.2
        )
        
        # Chain the operators in sequence
        super().__init__(operators=[self.ensemble, self.judge, self.verifier])
    
    def forward(self, *, inputs):
        # First, run the ensemble to get multiple answers
        ensemble_result = self.ensemble(inputs=inputs)
        
        # Next, judge the ensemble responses
        judge_result = self.judge(inputs={
            "query": inputs["query"],
            "responses": ensemble_result.responses
        })
        
        # Finally, verify the judge's answer
        verifier_result = self.verifier(inputs={
            "query": inputs["query"],
            "candidate_answer": judge_result.final_answer
        })
        
        return verifier_result

# Create and use the optimized pipeline
pipeline = RobustQAPipeline()
result = pipeline(inputs={"query": "What is the speed of light?"})

print(f"Verdict: {result.verdict}")
print(f"Final answer: {result.revised_answer}")
print(f"Explanation: {result.explanation}")
```

### Multi-model Pipeline

Combine different model types with automatic parallelization:

```python
from ember.core.non import VariedEnsemble, JudgeSynthesis
from ember.core.registry.model.model_module import LMModuleConfig
from ember.xcs.engine.execution_options import execution_options

# Define varied model configurations
model_configs = [
    LMModuleConfig(model_name="openai:gpt-4o", temperature=0.3),
    LMModuleConfig(model_name="anthropic:claude-3-haiku", temperature=0.4),
    LMModuleConfig(model_name="openai:gpt-4o-mini", temperature=0.5)
]

# Create operators
varied_ensemble = VariedEnsemble(model_configs=model_configs)
judge = JudgeSynthesis(model_name="anthropic:claude-3-sonnet")

# Execute with auto-parallelization
with execution_options(scheduler="wave", max_workers=len(model_configs)):
    # Get responses from the diverse models
    ensemble_result = varied_ensemble(inputs={"query": "Explain quantum computing."})
    
    # Synthesize into a single answer
    final_result = judge(inputs={
        "query": "Explain quantum computing.",
        "responses": ensemble_result.responses
    })

print(f"Final answer: {final_result.final_answer}")
```

## Creating Custom NON Operators

Create specialized operators that follow Ember's patterns:

```python
from typing import ClassVar, Type
from ember.core.non import UniformEnsemble
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel

class FactCheckerInput(EmberModel):
    """Input for the fact checker operator."""
    query: str
    domain: str

class FactCheckerOutput(EmberModel):
    """Output from the fact checker operator."""
    facts: list[str]
    sources: list[str]
    confidence: float

class FactCheckerSpecification(Specification):
    """Specification for the FactChecker operator."""
    input_model: Type[EmberModel] = FactCheckerInput
    structured_output: Type[EmberModel] = FactCheckerOutput
    
    prompt_template = """Check the following statement for factual accuracy in the {domain} domain.
    Statement: {query}
    
    Provide a list of verified facts and their sources.
    """

class FactChecker(Operator[FactCheckerInput, FactCheckerOutput]):
    """Custom operator for domain-specific fact checking."""
    
    # Class-level specification
    specification: ClassVar[Specification] = FactCheckerSpecification()
    
    # Instance attributes
    ensemble: UniformEnsemble
    confidence_threshold: float
    
    def __init__(self, *, confidence_threshold: float = 0.7):
        self.ensemble = UniformEnsemble(
            num_units=3,
            model_name="openai:gpt-4o",
            temperature=0.2
        )
        self.confidence_threshold = confidence_threshold
    
    def forward(self, *, inputs: FactCheckerInput) -> FactCheckerOutput:
        # Generate responses with the ensemble
        ensemble_result = self.ensemble(inputs={"query": inputs.query})
        
        # Process responses to extract facts and sources
        # (implementation details omitted for brevity)
        facts = ["Earth orbits the Sun", "A day on Earth is approximately 24 hours"]
        sources = ["Astronomy textbook", "NASA website"]
        confidence = 0.95
        
        return FactCheckerOutput(
            facts=facts,
            sources=sources,
            confidence=confidence
        )

# Use the custom operator
fact_checker = FactChecker(confidence_threshold=0.8)
result = fact_checker(inputs={"query": "The Earth orbits the Sun", "domain": "astronomy"})
```

## Related Documentation

- [Operators](operators.md) - Building custom computation units
- [Model Registry](model_registry.md) - Managing LLM configurations
- [Enhanced JIT](../xcs/JIT_OVERVIEW.md) - Optimized tracing and execution