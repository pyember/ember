# Ember Operators - Quickstart Guide

This guide introduces Ember's operator system - the core building blocks for constructing AI workflows. Operators in Ember are conceptually similar to PyTorch modules, providing a flexible and composable way to build complex processing pipelines.

## 1. Operator Fundamentals

Operators are stateful, typed computation units that:
- Take structured inputs and produce structured outputs
- Can be composed into complex graphs
- Are automatically parallelizable when possible
- Provide clear typing guarantees with Pydantic models

```python
from ember.core.registry.operator.base import Operator
from pydantic import BaseModel
from typing import List

# Define input/output types
class ClassifierInput(BaseModel):
    text: str
    categories: List[str]

class ClassifierOutput(BaseModel):
    category: str
    confidence: float
```

## 2. Creating a Basic Operator

```python
from ember.core.registry.model.model_module import LMModule, LMModuleConfig
from ember.core.registry.prompt_signature import Signature

# Define the signature (input/output schema + prompt template)
class ClassifierSignature(Signature):
    input_model = ClassifierInput
    structured_output = ClassifierOutput
    prompt_template = """Classify the following text into one of these categories: {categories}
    
Text: {text}

Respond with a JSON object with two keys:
- "category": The best matching category
- "confidence": A number between 0 and 1 indicating confidence
"""

# Create the operator
class TextClassifierOperator(Operator[ClassifierInput, ClassifierOutput]):
    signature = ClassifierSignature()
    
    def __init__(self, model_name: str = "openai:gpt-4o", temperature: float = 0.0):
        self.lm_module = LMModule(LMModuleConfig(
            model_name=model_name,
            temperature=temperature,
            response_format={"type": "json_object"}
        ))
    
    def forward(self, *, inputs: ClassifierInput) -> ClassifierOutput:
        # Render prompt from input
        prompt = self.signature.render_prompt(inputs=inputs)
        
        # Call LLM
        response = self.lm_module(prompt)
        
        # Parse JSON response
        try:
            import json
            result = json.loads(response)
            return ClassifierOutput(
                category=result["category"],
                confidence=result["confidence"]
            )
        except Exception as e:
            # Error handling
            raise ValueError(f"Failed to parse LLM response: {e}")
```

## 3. Using Operators

```python
# Instantiate the operator
classifier = TextClassifierOperator(model_name="anthropic:claude-3-haiku")

# Create input
input_data = ClassifierInput(
    text="The sky is blue and the sun is shining brightly today.",
    categories=["weather", "politics", "technology", "sports"]
)

# Execute the operator
result = classifier(inputs=input_data)

print(f"Category: {result.category}, Confidence: {result.confidence:.2f}")
```

## 4. Composition Patterns

### Sequential Composition

```python
from ember.core.non import Sequential

# Create a pipeline of operators
pipeline = Sequential(operators=[
    preprocessor,
    classifier,
    postprocessor
])

# Execute the entire pipeline with a single call
final_result = pipeline(inputs=initial_input)
```

### Parallel Composition (via graphs)

```python
from ember.xcs.graph import XCSGraph
from ember.xcs.engine import execute_graph

# Create operators
sentiment = SentimentAnalysisOperator(model_name="openai:gpt-4o")
classifier = TextClassifierOperator(model_name="anthropic:claude-3-haiku")
summarizer = TextSummarizerOperator(model_name="openai:gpt-4o-mini")

# Build graph
graph = XCSGraph()
graph.add_node(operator=sentiment, node_id="sentiment")
graph.add_node(operator=classifier, node_id="classifier")
graph.add_node(operator=summarizer, node_id="summarizer")
graph.add_node(operator=aggregator, node_id="aggregator")

# Define data flows
graph.add_edge(from_id="sentiment", to_id="aggregator")
graph.add_edge(from_id="classifier", to_id="aggregator")
graph.add_edge(from_id="summarizer", to_id="aggregator")

# Execute with automatic parallelization
result = execute_graph(
    graph=graph,
    global_input={"text": "Your input text here"},
    max_workers=3
)
```

## 5. Built-in Operators

Ember provides pre-built operators for common tasks:

```python
from ember.core.non import (
    UniformEnsemble,  # Run multiple instances of the same model
    VariedEnsemble,   # Run different models/configs in parallel
    JudgeSynthesis,   # Evaluate and combine multiple responses
    MostCommon,       # Select the most common answer from a set
    Verifier          # Verify and potentially correct an answer
)

# Create an ensemble of models
ensemble = UniformEnsemble(
    num_units=3,
    model_name="openai:gpt-4o", 
    temperature=0.7
)

# Create a judge to evaluate and combine responses
judge = JudgeSynthesis(model_name="anthropic:claude-3-sonnet")

# Execute them in a pipeline
responses = ensemble(inputs={"query": "What is the capital of France?"})
final_result = judge(inputs={"query": "What is the capital of France?", "responses": responses.responses})
```

## 6. Operator Best Practices

1. **Type Everything**: Use Pydantic models for all inputs and outputs
2. **Error Handling**: Robustly handle parsing errors and LLM failures
3. **Pure Forward Methods**: Keep `forward()` methods deterministic
4. **Reuse Operators**: Compose small, specialized operators for complex tasks
5. **Flexible Initialization**: Allow customization via constructor parameters
6. **Stateless When Possible**: Prefer immutable state after initialization

## 7. Advanced: Custom Operator Transformations

```python
from ember.xcs.tracer import jit

# Create a decorator to retry operations on failure
def with_retry(max_attempts=3):
    def decorator(op_class):
        class RetryOperator(op_class):
            def forward(self, *, inputs):
                for attempt in range(max_attempts):
                    try:
                        return super().forward(inputs=inputs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise
                        print(f"Attempt {attempt+1} failed, retrying...")
        return RetryOperator
    return decorator

# Apply the decorator
@with_retry(max_attempts=3)
class ReliableClassifier(TextClassifierOperator):
    pass

# Create a JIT-compiled pipeline with the reliable classifier
@jit
def analysis_pipeline(text, categories):
    reliable_op = ReliableClassifier()
    result = reliable_op(inputs={"text": text, "categories": categories})
    return result
```

## Next Steps

Learn more about:
- [Prompt Signatures](prompt_signatures.md) - Type-safe prompt templating
- [Model Registry](model_registry.md) - Managing LLM configurations
- [NON Patterns](non.md) - Networks of Networks composition
- [XCS Graphs](../advanced/xcs_graphs.md) - Advanced parallel execution