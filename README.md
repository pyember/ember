# Ember: Compositional Framework for Compound AI Systems

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Ember is a powerful, extensible Python framework for building and orchestrating Compound AI Systems and "Networks of Networks" (NONs). Inspired by the best of PyTorch, JAX, and other modern frameworks, Ember provides a familiar yet powerful API for composing complex AI systems with optimized execution.

**Core Features:**
- 🔥 **Eager Execution by Default**: Intuitive development experience with immediate execution results
- 🚀 **Parallel Graph Execution**: Automatic optimization of execution flow for concurrent operations
- 🧩 **Composable Operators**: PyTorch-like modular components for high reusability
- 🔌 **Extensible Registry System**: First-class support for custom models, operators, and evaluators
- ⚡ **Enhanced JIT System**: JAX-inspired tracing and graph optimization with hierarchical awareness
- 📊 **Built-in Evaluation**: Comprehensive tools for measuring and analyzing model performance
- 🔄 **Powerful Data Handling**: Extensible data pipeline integration with popular datasets

## Quick Installation

```bash
# Install using pip
pip install ember-ai

# Or using Poetry
poetry add ember-ai
```

## Motivating Example: Sophisticated Multi-LLM Architecture

Let's start with a real-world compound AI system that demonstrates Ember's power - a complex reasoning system with model ensembling, verification, and synthesis, all automatically parallelized:

```python
from typing import ClassVar
from ember.xcs.tracer import jit
from ember.xcs.engine import execution_options
from ember.core import non
from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel

# Define structured model inputs/outputs
class QueryInput(EmberModel):
    query: str
    
class ReasoningOutput(EmberModel):
    final_answer: str
    confidence: float

class ReasoningSpecification(Specification):
    input_model = QueryInput
    output_model = ReasoningOutput

@jit
class AdvancedReasoningSystem(Operator[QueryInput, ReasoningOutput]):
    """A sophisticated reasoning system with:
    1. Parallel LLM reasoning with different models
    2. Verification of reasoning steps
    3. Synthesis of final response
    """
    
    # Class-level specification declaration
    specification: ClassVar[Specification] = ReasoningSpecification()
    
    # Class-level field declarations with types
    reasoning_ensemble: non.UniformEnsemble
    verifier: non.Verifier
    synthesizer: non.JudgeSynthesis
    
    def __init__(self):
        # Initialize declared fields
        self.reasoning_ensemble = non.UniformEnsemble(
            num_units=3, 
            model_name="openai:gpt-4o",
            temperature=0.7
        )
        
        self.verifier = non.Verifier(
            model_name="anthropic:claude-3-sonnet",
            verification_criteria=["factual_accuracy", "coherence", "completeness"]
        )
        
        self.synthesizer = non.JudgeSynthesis(
            model_name="anthropic:claude-3-opus",
            temperature=0.2
        )
    
    def forward(self, *, inputs: QueryInput) -> ReasoningOutput:
        # Step 1: Get multiple reasoning paths (executed in parallel)
        reasoning_results = self.reasoning_ensemble(inputs={"query": inputs.query})
        
        # Step 2: Verify each reasoning path (also executed in parallel)
        verification_results = self.verifier(inputs={
            "query": inputs.query,
            "responses": reasoning_results["responses"]
        })
        
        # Step 3: Synthesize a final response from verified reasoning
        synthesis = self.synthesizer(inputs={
            "query": inputs.query,
            "verified_responses": verification_results["verified_responses"],
            "verification_scores": verification_results["scores"]
        })
        
        # Return structured output
        return ReasoningOutput(
            final_answer=synthesis["final_answer"],
            confidence=synthesis.get("confidence", 0.0)
        )

# Create and use the system (graph is automatically built and optimized)
reasoner = AdvancedReasoningSystem()

# Enable maximum parallelization
with execution_options(max_workers=4):
    # Using kwargs format for cleaner input
    result = reasoner(inputs={"query": "What are the economic implications of quantum computing?"})
    
print(f"Final synthesized answer: {result.final_answer}")
print(f"Confidence: {result.confidence:.2f}")
```

This example demonstrates how Ember automatically:
1. Traces the execution flow to build an optimized computation graph
2. Identifies parallel execution opportunities
3. Handles dependencies between operators correctly
4. Provides a clean, Pythonic API for complex AI system composition

## Progression of Examples

### 1. Minimal Example: Single LLM Operator

The simplest possible Ember application - a single LLM call wrapped in an operator:

```python
from typing import ClassVar
from ember.core.registry.operator.base import Operator
from ember.core.registry.model.model_module import LMModule, LMModuleConfig
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel

class SimpleInput(EmberModel):
    query: str
    
class SimpleOutput(EmberModel):
    answer: str
    
class SimpleQASpecification(Specification):
    input_model = SimpleInput
    output_model = SimpleOutput
    prompt_template = "Please answer this question: {query}"

class SimpleQA(Operator[SimpleInput, SimpleOutput]):
    """Basic question-answering operator using a single LLM."""
    
    # Class-level specification declaration
    specification: ClassVar[Specification] = SimpleQASpecification()
    
    # Class-level field declaration
    lm: LMModule
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        # Initialize the declared field
        self.lm = LMModule(LMModuleConfig(model_name=model_name))
        
    def forward(self, *, inputs: SimpleInput) -> SimpleOutput:
        # Generate prompt using specification
        prompt = self.specification.render_prompt(inputs)
        
        # Get response from LLM
        response = self.lm(prompt)
        
        # Return structured output
        return SimpleOutput(answer=response.strip())

# Create and use the operator with kwargs format
qa_system = SimpleQA()
result = qa_system(inputs={"query": "What is the capital of France?"})
print(result.answer)
```

### 2. Medium Complexity: Multi-Model Ensemble with JIT

A more sophisticated example with multiple models and automatic optimization:

```python
from typing import ClassVar, List
from ember.core.registry.operator.base import Operator
from ember.xcs.tracer import jit
from ember.core import non
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel

class EnsembleInput(EmberModel):
    query: str
    
class EnsembleOutput(EmberModel):
    final_answer: str
    confidence: float
    model_responses: List[str]
    
class EnsembleSpecification(Specification):
    input_model = EnsembleInput
    output_model = EnsembleOutput

@jit
class QAEnsemble(Operator[EnsembleInput, EnsembleOutput]):
    """Question-answering ensemble with multiple models and automatic graph building."""
    
    # Class-level specification declaration
    specification: ClassVar[Specification] = EnsembleSpecification()
    
    # Class-level field declarations
    ensemble: non.UniformEnsemble
    aggregator: non.MostCommon
    
    def __init__(self, num_models: int = 3):
        # Initialize declared fields
        self.ensemble = non.UniformEnsemble(
            num_units=num_models, 
            model_name="openai:gpt-4o-mini",
            temperature=0.7
        )
        
        self.aggregator = non.MostCommon()
    
    def forward(self, *, inputs: EnsembleInput) -> EnsembleOutput:
        # Get multiple responses from ensemble (automatically parallelized)
        ensemble_result = self.ensemble(inputs={"query": inputs.query})
        responses = ensemble_result["responses"]
        
        # Aggregate responses to get final answer
        aggregator_inputs = {
            "query": inputs.query,
            "responses": responses
        }
        final_result = self.aggregator(inputs=aggregator_inputs)
        
        # Return structured output
        return EnsembleOutput(
            final_answer=final_result["final_answer"],
            confidence=final_result.get("confidence", 0.0),
            model_responses=responses
        )

# Create and use the ensemble with kwargs format
ensemble = QAEnsemble(num_models=5)
result = ensemble(inputs={"query": "What is the tallest mountain in the world?"})
print(f"Answer: {result.final_answer}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Number of model responses: {len(result.model_responses)}")
```

### 3. Advanced Example: Complex Pipeline with Multiple Stages

Demonstrates a multi-stage pipeline with different operators and automatic execution optimization:

```python
from typing import ClassVar
from ember.core.registry.operator.base import Operator
from ember.xcs.tracer import jit, execution_options
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.core.registry.model.model_module import LMModule, LMModuleConfig

class DocumentInput(EmberModel):
    document: str
    max_words: int = 150

class DocumentOutput(EmberModel):
    summary: str
    analysis: str
    fact_check: str

class DocumentProcessorSpecification(Specification):
    input_model = DocumentInput
    output_model = DocumentOutput
    prompt_template = """Summarize the following text in {max_words} words or less:
    
{document}
    
Summary:"""

@jit
class DocumentProcessor(Operator[DocumentInput, DocumentOutput]):
    """Advanced document processing pipeline with multiple stages."""
    
    # Class-level specification declaration
    specification: ClassVar[Specification] = DocumentProcessorSpecification()
    
    # Class-level field declarations
    summarizer: LMModule
    analyzer: LMModule
    fact_checker: LMModule
    
    def __init__(self):
        # Initialize declared fields
        self.summarizer = LMModule(
            LMModuleConfig(model_name="anthropic:claude-3-sonnet")
        )
        
        self.analyzer = LMModule(
            LMModuleConfig(model_name="openai:gpt-4o")
        )
        
        self.fact_checker = LMModule(
            LMModuleConfig(model_name="anthropic:claude-3-haiku")
        )
    
    def forward(self, *, inputs: DocumentInput) -> DocumentOutput:
        # Stage 1: Create a summary
        summary_prompt = self.specification.render_prompt(inputs)
        summary = self.summarizer(summary_prompt)
        
        # Stage 2: Analyze themes (runs in parallel with fact checking)
        analysis_prompt = f"Analyze the key themes in this text: {summary}"
        analysis_task = self.analyzer(analysis_prompt)
        
        # Stage 3: Fact check (runs in parallel with analysis)
        fact_check_prompt = f"Identify any potential factual errors in: {summary}"
        fact_check_task = self.fact_checker(fact_check_prompt)
        
        # Results are automatically awaited when accessed
        return DocumentOutput(
            summary=summary,
            analysis=analysis_task,
            fact_check=fact_check_task
        )

# Process a document with automatic parallelization
processor = DocumentProcessor()

with execution_options(max_workers=3):
    # Using kwargs format for input
    result = processor(inputs={
        "document": "Long document text...",
        "max_words": 200
    })

print(f"Summary: {result.summary}")
print(f"Analysis: {result.analysis}")
print(f"Fact Check: {result.fact_check}")
```

## Core Components

### Model Registry

Ember's Model Registry provides a unified interface to various LLM providers with comprehensive management features:

```python
from ember import initialize_ember
from ember.core.registry.model import ModelEnum
from ember.core.registry.model.base.services import ModelService, UsageService

# One-line initialization (quickest approach)
service = initialize_ember(usage_tracking=True)

# Use models directly with service (like a function)
response = service(ModelEnum.OPENAI_GPT4O, "What is the capital of France?")
print(response.data)

# Or with more explicit API
response = service.invoke_model(
    model_id=ModelEnum.ANTHROPIC_CLAUDE_3_SONNET,
    prompt="Explain quantum computing in simple terms."
)

# Track usage statistics
usage = service.usage_service.get_total_usage()
print(f"Total tokens used: {usage.tokens}")
print(f"Estimated cost: ${usage.cost:.6f}")
```

### Operators

Operators are the fundamental building blocks in Ember, similar to PyTorch's `nn.Module`. Here's a more comprehensive example showing proper typing and initialization:

```python
from typing import ClassVar, List
from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.core.registry.model.model_module import LMModule, LMModuleConfig

# Define structured I/O models for type safety
class SentimentInput(EmberModel):
    text: str

class SentimentOutput(EmberModel):
    sentiment: str
    confidence: float
    reasons: List[str]

# Create a specification with structured I/O and prompt template
class SentimentSpecification(Specification):
    input_model = SentimentInput
    output_model = SentimentOutput
    prompt_template = """Analyze the sentiment of the following text:
    
{text}
    
Provide your analysis as a JSON object with fields:
- sentiment: either "positive", "negative", or "neutral"
- confidence: a float between 0 and 1
- reasons: a list of key phrases that support your analysis"""

# Create the operator
class SentimentAnalyzer(Operator[SentimentInput, SentimentOutput]):
    # Class-level specification declaration
    specification: ClassVar[Specification] = SentimentSpecification()
    
    # Class-level field declaration
    lm: LMModule
    
    def __init__(self, model_name: str = "openai:gpt-4o"):
        # Initialize the declared field
        self.lm = LMModule(LMModuleConfig(model_name=model_name))
        
    def forward(self, *, inputs: SentimentInput) -> SentimentOutput:
        # Render prompt from inputs using the specification
        prompt = self.specification.render_prompt(inputs)
        
        # Get response from LLM
        response = self.lm(prompt)
        
        # Parse structured output using the specification
        return self.specification.parse_response(response)

# Use the operator with kwargs format
analyzer = SentimentAnalyzer()
result = analyzer(inputs={"text": "I absolutely loved the new movie!"})
print(f"Sentiment: {result.sentiment} (Confidence: {result.confidence:.2f})")
print(f"Reasons: {', '.join(result.reasons)}")
```

### Networks of Networks (NON)

Ember provides powerful components for building complex Networks of Networks (NON):

```python
from typing import ClassVar
from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.core import non

class NetworkInput(EmberModel):
    query: str
    
class NetworkOutput(EmberModel):
    final_answer: str
    
class SubNetworkSpecification(Specification):
    input_model = NetworkInput
    output_model = NetworkOutput

class SubNetwork(Operator[NetworkInput, NetworkOutput]):
    """SubNetwork that composes an ensemble with verification."""
    
    # Class-level specification declaration
    specification: ClassVar[Specification] = SubNetworkSpecification()
    
    # Class-level field declarations
    ensemble: non.UniformEnsemble
    verifier: non.Verifier
    
    def __init__(self):
        # Initialize declared fields
        self.ensemble = non.UniformEnsemble(
            num_units=2, model_name="openai:gpt-4o", temperature=0.0
        )
        self.verifier = non.Verifier(model_name="openai:gpt-4o", temperature=0.0)
    
    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        # Process through ensemble
        ensemble_result = self.ensemble(inputs={"query": inputs.query})
        
        # Get the candidate answer
        candidate_answer = ensemble_result.get("final_answer", "")
        
        # Verify the ensemble's output
        verification_input = {
            "query": inputs.query,
            "candidate_answer": candidate_answer,
        }
        verified_result = self.verifier(inputs=verification_input)
        
        # Return structured output
        return NetworkOutput(final_answer=verified_result.get("final_answer", ""))

# Create nested networks
class NestedNetwork(Operator[NetworkInput, NetworkOutput]):
    """Nested network with sub-networks and final judgment."""
    
    # Class-level specification declaration
    specification: ClassVar[Specification] = SubNetworkSpecification()
    
    # Class-level field declarations
    sub1: SubNetwork
    sub2: SubNetwork
    judge: non.JudgeSynthesis
    
    def __init__(self):
        # Initialize declared fields
        self.sub1 = SubNetwork()
        self.sub2 = SubNetwork()
        self.judge = non.JudgeSynthesis(model_name="openai:gpt-4o", temperature=0.0)
    
    def forward(self, *, inputs: NetworkInput) -> NetworkOutput:
        # Process through parallel sub-networks
        s1_out = self.sub1(inputs=inputs)
        s2_out = self.sub2(inputs=inputs)
        
        # Synthesize results using the judge
        judge_inputs = {
            "query": inputs.query, 
            "responses": [s1_out.final_answer, s2_out.final_answer]
        }
        judged_result = self.judge(inputs=judge_inputs)
        
        # Return structured output
        return NetworkOutput(final_answer=judged_result.get("final_answer", ""))

# Use the nested network with kwargs format
network = NestedNetwork()
result = network(inputs={"query": "Explain quantum computing"})
print(f"Final answer: {result.final_answer}")
```

### Enhanced JIT and AutoGraph

Ember provides a powerful tracing system that automatically builds and optimizes execution graphs with operators:

```python
from typing import ClassVar, Dict, List
from ember.xcs.tracer import jit
from ember.xcs.engine import execution_options  
from ember.core import non
from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel

# Define our input and output types
class PipelineInput(EmberModel):
    query: str

class PipelineOutput(EmberModel):
    answer: str

class PipelineSpecification(Specification):
    input_model = PipelineInput
    output_model = PipelineOutput

@jit
class ComplexPipeline(Operator[PipelineInput, PipelineOutput]):
    """An operator-based pipeline with automatic graph building and optimization."""
    
    # Class-level specification declaration
    specification: ClassVar[Specification] = PipelineSpecification()
    
    # Class-level field declarations
    ensemble: non.UniformEnsemble
    judge: non.JudgeSynthesis
    
    def __init__(self):
        # Initialize declared fields
        self.ensemble = non.UniformEnsemble(
            num_units=3, 
            model_name="openai:gpt-4o"
        )
        
        self.judge = non.JudgeSynthesis(
            model_name="anthropic:claude-3-sonnet"
        )
    
    def forward(self, *, inputs: PipelineInput) -> PipelineOutput:
        # This execution flow is automatically traced and optimized
        ensemble_result = self.ensemble(inputs={"query": inputs.query})
        
        judge_inputs = {
            "query": inputs.query,
            "responses": ensemble_result["responses"]
        }
        
        final_result = self.judge(inputs=judge_inputs)
        return PipelineOutput(answer=final_result["final_answer"])

# Create pipeline operator
pipeline = ComplexPipeline()

# First call traces and builds the graph with kwargs format
with execution_options(max_workers=3):
    result = pipeline(inputs={"query": "Explain the theory of relativity"})
    print(f"Result: {result.answer}")

# Subsequent calls reuse the optimized graph
result = pipeline(inputs={"query": "What is quantum entanglement?"})
print(f"Result: {result.answer}")
```

### Data Handling and Evaluation

Ember includes powerful tools for dataset handling and model evaluation:

```python
from ember.core.utils.data import DataLoader, DataTransformer
from ember.core.utils.eval import EvaluationPipeline, Evaluator

# Load a dataset
loader = DataLoader.from_registry("mmlu")
dataset = loader.load(subset="physics", split="test")

# Transform data for evaluation
transformer = DataTransformer()
prepared_data = transformer.transform(dataset, target_format="qa_pairs")

# Create an evaluation pipeline
eval_pipeline = EvaluationPipeline([
    Evaluator.from_registry("accuracy"),
    Evaluator.from_registry("response_quality")
])

# Define the model to evaluate
from ember.core.registry.model.model_module import LMModule, LMModuleConfig
model = LMModule(LMModuleConfig(model_name="openai:gpt-4o"))

# Run evaluation
results = eval_pipeline.evaluate(model, prepared_data)
print(f"Accuracy: {results['accuracy']:.2f}")
print(f"Response Quality: {results['response_quality']:.2f}")
```

## Use Cases

Ember is well-suited for complex AI pipelines such as:

- **Multi-model ensembling** for improved answer quality and robustness
- **Verification and self-correction** pipelines with structured workflows
- **Multi-agent systems** with specialized roles and coordination
- **Tool-augmented systems** combining LLMs with specialized tools
- **Complex reasoning chains** with intermediate evaluations and refinement
- **Evaluation and benchmarking** of compound AI systems at scale

## Documentation

For comprehensive documentation, including tutorials, API reference, and advanced usage:

- [Ember Documentation](https://pyember.ai)
- [Architecture Overview](ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Examples Directory](src/ember/examples)

### Quickstart Guides

- [Model Registry Quickstart](docs/quickstart/model_registry.md) - Get started with LLM integration
- [Operators Quickstart](docs/quickstart/operators.md) - Build reusable computation units
- [Prompt Specifications Quickstart](docs/quickstart/prompt_specifications.md) - Type-safe prompt engineering
- [NON Quickstart](docs/quickstart/non.md) - Networks of Networks patterns
- [Data Quickstart](docs/quickstart/data.md) - Working with datasets and evaluation

## Performance

Ember is designed for high-performance LLM orchestration:

- **Automatic parallelization** of independent operations
- **Hierarchical operator awareness** for optimal execution planning
- **Minimal overhead** compared to direct API calls
- **Smart caching** of repeated operations and sub-graphs
- **Efficient resource utilization** with dynamic scheduling

## Community

- [GitHub Issues](https://github.com/pyember/ember/issues): Bug reports and feature requests
- [GitHub Discussions](https://github.com/pyember/ember/discussions): Questions and community support
- [Discord](https://discord.gg/ember-ai): Join our community chat

## License

Ember is released under the [Apache 2.0 License](LICENSE).

---

Built with 🔥 by the Ember community