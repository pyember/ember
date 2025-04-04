# Ember Operators

Operators provide a typed, composable abstraction for building AI workflows. This guide introduces core concepts and usage patterns.

## 1. Core Concepts

Operators are:
- Typed computation units with structured inputs/outputs
- Composable into complex processing graphs
- Automatically parallelizable
- Thread-safe through immutable design

```python
from typing import List
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.types.ember_model import EmberModel

class ClassifierInput(EmberModel):
    text: str
    categories: List[str]

class ClassifierOutput(EmberModel):
    category: str
    confidence: float
```

## 2. Basic Implementation

```python
from typing import ClassVar, Type
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.specification.specification import Specification

class ClassifierSpecification(Specification):
    input_model: Type[EmberModel] = ClassifierInput
    structured_output: Type[EmberModel] = ClassifierOutput
    prompt_template: str = """Classify the following text into one of these categories: {categories}
    
Text: {text}

Respond with a JSON object with two keys:
- "category": The best matching category
- "confidence": A number between 0 and 1 indicating confidence
"""

class TextClassifierOperator(Operator[ClassifierInput, ClassifierOutput]):
    specification: ClassVar[Specification] = ClassifierSpecification()
    lm_module: LMModule
    
    def __init__(self, *, model_name: str = "openai:gpt-4o", temperature: float = 0.0) -> None:
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
        )
    
    def forward(self, *, inputs: ClassifierInput) -> ClassifierOutput:
        prompt = self.specification.render_prompt(inputs=inputs)
        response = self.lm_module(prompt=prompt)
        
        try:
            import json
            result = json.loads(response)
            return ClassifierOutput(
                category=result["category"],
                confidence=result["confidence"]
            )
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
```

## 3. Invocation

```python
# Instantiate
classifier = TextClassifierOperator(model_name="anthropic:claude-3-haiku")

# Execute with dict-style inputs
result = classifier(inputs={
    "text": "The sky is blue and the sun is shining brightly today.",
    "categories": ["weather", "politics", "technology", "sports"]
})

print(f"Category: {result.category}, Confidence: {result.confidence:.2f}")
```

## 4. Composition Patterns

### Sequential

```python
from ember.core import non

pipeline = non.Sequential(operators=[preprocessor, classifier, postprocessor])
result = pipeline(inputs={"text": "Your input text here"})
```

### Parallel

```python
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine.unified_engine import execute_graph

# Setup operators
sentiment = SentimentAnalysisOperator(model_name="openai:gpt-4o")
classifier = TextClassifierOperator(model_name="anthropic:claude-3-haiku")
summarizer = TextSummarizerOperator(model_name="openai:gpt-4o-mini")

# Build computation graph
graph = XCSGraph()
graph.add_node(operator=sentiment, node_id="sentiment")
graph.add_node(operator=classifier, node_id="classifier")
graph.add_node(operator=summarizer, node_id="summarizer")
graph.add_node(operator=aggregator, node_id="aggregator")

# Define dependencies
graph.add_edge(from_id="sentiment", to_id="aggregator")
graph.add_edge(from_id="classifier", to_id="aggregator")
graph.add_edge(from_id="summarizer", to_id="aggregator")

# Execute with parallelization
result = execute_graph(
    graph=graph,
    global_input={"text": "Your input text here"},
    max_workers=3
)
```

### Execution Control

```python
from ember.xcs.engine.execution_options import execution_options

with execution_options(scheduler="parallel", max_workers=4):
    result = pipeline(inputs={"query": "What is the capital of France?"})
```

## 5. Built-in Operators

```python
from ember.core import non

# Ensemble with multiple identical models
ensemble = non.UniformEnsemble(
    num_units=3,
    model_name="openai:gpt-4o", 
    temperature=0.7
)

# Meta-reasoning synthesis 
judge = non.JudgeSynthesis(model_name="anthropic:claude-3-sonnet")

# Execution pipeline
ensemble_result = ensemble(inputs={"query": "What is the capital of France?"})
final_result = judge(inputs={
    "query": "What is the capital of France?", 
    "responses": ensemble_result.responses
})
```

## 6. JIT and Transformations

```python
from typing import ClassVar
from ember.xcs.jit import jit
from ember.xcs.transforms.vmap import vmap
from ember.xcs.transforms.pmap import pmap

# JIT optimization
@jit(sample_input={"text": "Sample", "categories": ["a", "b"]})
class OptimizedOperator(Operator[ClassifierInput, ClassifierOutput]):
    specification: ClassVar[Specification] = ClassifierSpecification()
    
    def forward(self, *, inputs: ClassifierInput) -> ClassifierOutput:
        # Implementation
        return ClassifierOutput(category="category", confidence=0.95)

# Container pattern
@jit
class Pipeline(Operator[ProcessingInput, ProcessingOutput]):
    specification: ClassVar[Specification] = PipelineSpecification()
    preprocessor: PreprocessOperator
    classifier: OptimizedOperator
    
    def __init__(self, *, config_param: str = "default") -> None:
        self.preprocessor = PreprocessOperator(param=config_param)
        self.classifier = OptimizedOperator()
    
    def forward(self, *, inputs: ProcessingInput) -> ProcessingOutput:
        intermediate = self.preprocessor(inputs=inputs)
        return self.classifier(inputs=intermediate)

# Vectorization
batch_processor = vmap(single_item_processor)
results = batch_processor(inputs=[input1, input2, input3])

# Parallelization
parallel_processor = pmap(compute_intensive_operator)
```

## 7. Best Practices

1. **Strong Typing**: Use EmberModel for all inputs/outputs with proper typing
2. **Class-Level Fields**: Declare fields with types at class level
3. **Immutable Design**: Keep state immutable after initialization 
4. **Named Parameters**: Use `__init__(*, param1, param2)` pattern
5. **ClassVar for Specifications**: Use `specification: ClassVar[Specification]`
6. **Dict-Style Inputs**: Use `inputs={"key": value}` format
7. **Clean Forward Methods**: Keep `forward()` methods pure and deterministic
8. **Error Handling**: Catch LLM failures with specific error messages
9. **Small Components**: Build small, focused operators for composition
10. **Validate Outputs**: Return properly typed model instances

## Next Steps

- [Prompt Specifications](prompt_signatures.md) - Type-safe templating
- [Model Registry](model_registry.md) - LLM configuration
- [NON Patterns](non.md) - Networks of Networks
- [XCS Overview](../xcs/README.md) - Computation system
- [JIT](../xcs/JIT_OVERVIEW.md) - Just-In-Time compilation
- [Execution Options](../xcs/EXECUTION_OPTIONS.md) - Execution control
- [Transforms](../xcs/TRANSFORMS.md) - Vectorization and parallelization