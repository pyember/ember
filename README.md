# Ember: Compositional framework for Compound AI Systems and "Networks of Networks" (NONs)

## 1. Introduction

### What is Ember?

Ember is a Python library for constructing Compound AI Systems and "Networks of Networks" (NONs). Ember aims to provide, for the **NON** construction context, a similar experience to PyTorch/FLAX in the **NN** context. 

The goal of Ember's design is to provide:

- **The structural and efficiency beniefits**: of JAX/XLA-like graph execution.  
- **The top-level compositional and eager-mode user experience**: of PyTorch/FLAX.

With Ember, we can compose complex pipelines (a.k.a. "Networks of Networks" architectures) and multi-model, multi-step systems, that both execute efficiently and are easy to read and understand.

**Why Ember?**
- **Eager Execution**: by default, which AI practitioners accustomed to NN architectures design may find more intuitive (ala PyTorch). 

- **Graph Scheduling** behind the scenes, allowing us to optionally run "fan-out" computations in parallel (ala JAX/XLA, and their transform and concurrency orientation).

- **Composable** with "Operators" reminiscent of PyTorch's `nn.Module`: you can chain, nest, and re-use them. 

- **Extensible**: via a registry-based approach for custom operators, model-APIs, datasets, evaluation functions, and eventual integrations.

### Real-World Use Cases

- **Best-of-K** multi-model inference-time consortiums, ala Are More LLM Calls Are All You Need (Chen et al., 2024), Networks of Networks (Davis et al., 2024), Compound AI Systems (Zaharia et al., 2024), and Learning to Reason with LLMs (OpenAI et al., 2024).
- **Multi-agent LLM-based CAIS pipelines**: such as STORM (Shao et al., 2024), etc

- add more... 

### 2. A Quick Example: Composing a Multi-Model Ensemble

Below is a short snippet illustrating Ember's API for assembling a multi-model ensemble.

```python
```

**What’s happening?**

- We fan out (via an ensemble operator) to three different LLMs, each receiving the same query.
- We then fan in their responses with a JudgeBasedOperator,” (ala `Networks of Networks`) which considers the 'inputs/advise of the ensemble members and outputs a final answer.
- Under the hood, Ember can run amenable calls in parallel if configured with a concurrency plan—no special code required beyond defining to_plan() for operators or wrapping Operators using a tracer with the @jit decorator.

# 3. Core Components of Ember

The building blocks of Ember are separated via a '**registry-based**' design for composability.
You can declare or register your providers and models, giving Ember information about rate-limits, costs, etc so it can optimize (more on this later). You can also declare or register new **Operators**, **Datasets**, **Evaluation Functions**, and **Prompts Signatures** in the `registry` directory.

## 3.1 Model Registry

The Model Registry centralizes all the model configurations and metadata. It's the backbone of Ember's approach to discovering and managing LLM endpoints (like **OpenAI**'s `gpt-4o`, **Anthropic**'s `claude-3.5-sonnet`, **Google**'s `gemini-1.5-pro`, proprietary and OSS models via **IBM WatsonX** or custom/OSS LLMs e.g. served via **Ember**'s serving endpoint's for OSS models like DBRX, Llama, Deepseek, Qwen, etc.)

### Key Features
- **Registration**: You define a ModelInfo object (containing metadata about the cost, rate limits, provider info, and an API key) and pass it to the registry.
- **Lookup**: You can retrieve models by strine ID or enum or easy referencing in your code. 
- **Usage Tracking**: Ember can track usage, cost, and latency logs for models and providers, and provide usage summaries.


#### Example: Registering and using a model

```python
# Initialize registry
registry = ModelRegistry()
usage_service = UsageService()
model_service = ModelService(registry, usage_service)

# Register an OpenAI GPT-4 style model
openai_info = ModelInfo(
    model_id="openai:gpt-4o",  # unique ID
    model_name="gpt-4o",
    cost=ModelCost(input_cost_per_thousand=0.03, output_cost_per_thousand=0.06),
    rate_limit=RateLimit(tokens_per_minute=80000, requests_per_minute=5000),
    provider=ProviderInfo(name="OpenAI", default_api_key=settings.openai_api_key),
    api_key=settings.openai_api_key,
)

registry.register_model(openai_info)

# Use it
resp = model_service("openai:gpt-4o", "Hello from Ember!")
print(resp.data)  # LLM response
```

## 3.2 LM Modules

### Concept

While the Model Registry is about storing info, LM Modules are the actual callable objects you use in an operator. Each LM Module:
- Wraps a specific model from the registry (e.g., “gpt-4o”).
- Provides a **unified** `.generate_response(prompt, **kwargs)` or direct `__call__` interface.
- Optionally sets generation parameters like temperature, max tokens, or persona.

Why Modules?
- Keep a **uniform calling convention** across many LLMs.
- **Integrate easily with Ember’s concurrency model** (fan-out calls can just invoke multiple LM Modules in parallel).

### Basic Pattern

```python
from src.ember.registry.operator.operator_base import LMModuleConfig, LMModule

config = LMModuleConfig(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=200,
)
lm = LMModule(config, model_registry=registry)

result = lm("What is the capital of France?")
print("Answer:", result)
```

### Where They Shine
- **Ensembles**: If you want to call the same model multiple times or different models in parallel, create multiple LM Modules.
- **Chain-of-thought**: If you want to do a multi-step reasoning workflow, you can pass partial outputs into subsequent LM calls, all orchestrated by your Ember operator logic.


# 4. Signatures and `SPECIFICATIONS`

### Why Signatures?

A Signature describes what inputs an operator expects and how its output should look. This concept, inspired by [DSPy’s structured specification approach](https://arxiv.org/abs/… or dspy.io), helps ensure your operators remain well-defined and type-safe.

### What They Include
	1.	required_inputs: A list of field names that must appear in the input dictionary.
	2.	prompt_template: An optional string template to generate LLM prompts. It can reference the required inputs (e.g., "The question is {question}. Summarize in one line.").
	3.	structured_output (optional): A Pydantic model for validating the operator’s output.
	4.	input_model (optional): A Pydantic model that ensures the operator’s input is well-formed.

### Example: A custom signature for network intrusion detection labeling

```python
class CaravanLabelsOutput(BaseModel):
    """Example output model for Caravan labeling."""

    label: str


class CaravanLabelingInputs(BaseModel):
    question: str


class CaravanLabelingSignature(Signature):
    """Signature for labeling network flows as benign or malicious."""

    required_inputs: List[str] = ["question"]
    prompt_template: str = (
        "You are a network security expert.\n"
        "Given these unlabeled flows:\n{question}\n"
        "Label each flow as 0 for benign or 1 for malicious, one per line, no explanation.\n"
    )
    structured_output: Optional[Type[BaseModel]] = CaravanLabelsOutput
    input_model: Optional[Type[BaseModel]] = CaravanLabelingInputs

```

### Benefits
- **Better debugging**: If you pass the wrong dict shape to an operator, you’ll know immediately.
- **Future expansions**: This paves the way for advanced transformations or auto-documentation.
- **DSPy synergy**: In the future, Ember can seamlessly import DSPy “specifications” as Signatures.

# 5. Operators 

Operators are the 'workhorse' components of Ember. If you're familiar with PyTorch, think of them like `nn.Module`s.

They:
- Encapsulate an **operator** (e.g., run an LM call, combine multiple output, invoke a tool or function, do text processing, etc.)
- May define or reference a **Signature** for typed I/O. 
- Can contain LMModules or other Operators as sub-components.
- Provide a .forward(inputs) for **eager** exeuction, and optionally a .to_plan(inputs) for concurrency or scheduling. 

## 5.1 Lifecycle of an Operator

1. Instatiation:
```python
op = NewOperator(lm_modules=[...], signature=)
```

2. (Optional) Input validation:
If there's a signature, Ember checks if the input matches input_model or required_inputs. 

3. Execution:
- forward (inputs): runs synchronously in python, returns immediate results. 
- to_plan (inputs): returns an ExecutionPlan if you want Ember's scheduler to handle concurrency and optimize execution. 
 

5. (Optional) Output validation:
If there's a structured_output in the signature, Ember attempts to cast/validate the result.


## 5.2 Common Operator Types

- **RECURRENT**: Shape-preserving operators that refine or iterate on inputs over multiple steps. E.g., a self-refinement operator that calls a model repeatedly until convergence
- **FAN_IN**: Dimensionality-reducing operators that combine, merge, or select amonst multiple inputs (e.g., aggregator, judge, verifier-based judge, rank-filter, or majority-vote). 
- **FAN_OUT**: Dimensionality-expanding operators that produce multiple outputs from a single input (e.g., ensemble operators, or calls that split or replicate an input, such as prompt mixin operators that take in a single prompt and remix it by adding prefixes, suffixes, etc DSPy optimizer search style).

## 5.3 Example: JudgeExplainerOperator
Takes in multiple responses in a structured input, passes those calls to a judge lm call, returns two outputs: a final answer and an explanation in a structured output with a signature / prompt that asks for an explanation and a parse method call or GetAnswer op call or something to get afinite MCQ-compatible output, or True/False. 

```python
from collections import Counter
from src.ember.registry.operator.operator_base import Operator

class JudgeExplainerOperator(Operator):

```
### Why This Matters:
It’s **short**, **explicit**, and **composable**. You can feed multiple model outputs into this operator to get a final “sythesized” verdict.


# 6. GraphExecution and Scheduling

While operators can run eagerly, Ember also enables **graph-based** orchestration:
- **NoNGraph** structures let you define a set of nodes (each node = an operator) and specify dependencies.
- Basic default: a **topological sort** ensures each node runs only after its inputs are ready.
- **Parallel execution** arises naturally for nodes that don’t depend on each other.

## 6.1 NoNGraph

A simple container that holds:
	•	A dictionary of named nodes.
	•	Each node references an Operator plus the names of other nodes whose outputs feed into it.

## 6.2 Execution Plan & Scheduler
- *ExecutionPlan* is a representation of tasks + dependencies. Operators that implement to_plan() produce these tasks.
- **Scheduler** runs the tasks in topological order, spawning parallel threads or processes where possible.

**Example**: If you have a single “EnsembleOperator” that fans out to N LM calls, to_plan() might generate N tasks, all depending on the single input. The scheduler can run them concurrently, then unify their results.

# 7. Unifying this all: Extending multi-model example

Let's demonstrate a bigger pipeline:

1. **Goal**: Query three different models (gemini-1.5-pro, claude-3.5-sonnet, and gpt-4o) in parallel, then have a “judge” operator (which itself might be an LM-based aggregator) decide on a final best answer.
2. **Setup**
```python
from src.ember.registry.model.model_registry import ModelRegistry
from src.ember.registry.operator.operator_registry import EnsembleOperator, GetAnswerOperator
from src.ember.registry.operator.operator_base import LMModuleConfig, LMModule

# 1) Register the models
registry = ModelRegistry()
# (Imagine we've done registry.register_model(...) for each: gemini, claude, gpt-4o, etc.)

# 2) Create LMModules for each
gemini_mod = LMModule(LMModuleConfig(model_name="gemini-1.5-pro"), registry)
claude_mod = LMModule(LMModuleConfig(model_name="claude-3.5-sonnet"), registry)
g4o_mod    = LMModule(LMModuleConfig(model_name="gpt-4o"), registry)

# 3) Instantiate an EnsembleOperator
ensemble_op = EnsembleOperator(lm_modules=[gemini_mod, claude_mod, g4o_mod])

# 4) Instantiate a "Judge" operator (GetAnswerOperator or MostCommonOperator)
#    Here let's assume "GetAnswerOperator" uses a 'final_judge' LMModule
judge_mod = LMModule(LMModuleConfig(model_name="o1-mini"), registry)
judge_op = ...
```

3.	Create a NoNGraph
```python
from src.ember.core.graph_executor import NoNGraphData, GraphExecutorService

graph_data = NoNGraphData()
# Node: "ensemble"
graph_data.add_node(
    name="ensemble", 
    operator=ensemble_op, 
    inputs=[]  # no prior node dependencies
)
# Node: "judge"
graph_data.add_node(
    name="judge",
    operator=judge_op,
    inputs=["ensemble"]  # feed ensemble output into judge
)

# 5) Provide the final input to the graph
input_data = {"query": "Explain how to set up Ember for multi-model parallel usage"}

# 6) Execute the graph
executor_service = GraphExecutorService()
results = executor_service.run(graph_data=graph_data, input_data=input_data)

print("Final answer from the judge:", results["judge"]["final_answer"])
```

4.	What Happens
	•	The “ensemble” node fans out to 3 different LLM modules concurrently.
	•	Once finished, their combined output (list of responses) flows to the “judge.”
	•	The “judge” operator uses its LM to pick or refine the best among them.
	•	Finally, we get a single consolidated string in results["judge"]["final_answer"].

Key Takeaways
	•	The code is explicit about how data flows.
	•	Ember automatically manages concurrency for the fan-out ensemble if you implement or use the operator’s to_plan() method.
	•	You can easily swap the aggregator with a “MostCommonOperator” or any custom logic.
	•	You can nest subgraphs or add more nodes as your pipeline grows in complexity.


# Next steps:

In the next sections, we'll cover installation, quickstart instructions, advancd topics (like DSPy-based signature integration and LiteLLM synergy), and details on contributing to Ember's ecosystem.

## Conclusion

Ember provides a compositional, extensible framework for building NoN compound AI systems. By following design principles akin to PyTorch, it offers a familiar and approachable structure for researchers and developers. Emphasizing explicit building blocks makes systems more understandable and manageable, even as they scale in complexity. With Ember, you can experiment with multiple models, operators, and consensus mechanisms, potentially improving the robustness and accuracy of AI-driven applications.

## License

Apache 2.0 License.


