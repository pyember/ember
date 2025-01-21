# Avior: Compositional framework for Compound AI Systems and "Networks of Networks" (NONs)

## 1. Introduction

### What is Avior?

Avior is a Python library for constructing Compound AI Systems and "Networks of Networks" (NONs). Avior aims to provide, for the **NON** construction context, a similar experience to PyTorch/FLAX in the **NN** context. 

The goal of Avior's design is to provide:

- **The structural and efficiency beniefits**: of JAX/XLA-like graph execution.  
- **The top-level compositional and eager-mode user experience**: of PyTorch/FLAX.

With Avior, we can compose complex pipelines (a.k.a. "Networks of Networks" architectures) and multi-model, multi-step systems, that both execute efficiently and are easy to read and understand.

**Why Avior?**
- **Eager Execution**: by default, which AI practitioners accustomed to NN architectures design may find more intuitive (ala PyTorch). 

- **Graph Scheduling** behind the scenes, allowing us to optionally run "fan-out" computations in parallel (ala JAX/XLA, and their transform and concurrency orientation).

- **Composable** with "Operators" reminiscent of PyTorch's `nn.Module`: you can chain, nest, and re-use them. 

- **Extensible**: via a registry-based approach for custom operators, model-APIs, datasets, evaluation functions, and eventual integrations.

### Real-World Use Cases

- **Best-of-K** multi-model inference-time consortiums, ala Are More LLM Calls Are All You Need (Chen et al., 2024), Networks of Networks (Davis et al., 2024), Compound AI Systems (Zaharia et al., 2024), and Learning to Reason with LLMs (OpenAI et al., 2024).
- **Multi-agent LLM-based CAIS pipelines**: such as STORM (Shao et al., 2024), etc

- add more... 

### 2. A Quick Example: Composing a Multi-Model Ensemble

Below is a short snippet illustrating Avior's API for assembling a multi-model ensemble.

```python
```

**What’s happening?**

- We fan out (via an ensemble operator) to three different LLMs, each receiving the same query.
- We then fan in their responses with a JudgeBasedOperator,” (ala `Networks of Networks`) which considers the 'inputs/advise of the ensemble members and outputs a final answer.
- Under the hood, Avior can run amenable calls in parallel if configured with a concurrency plan—no special code required beyond defining to_plan() for operators or wrapping Operators using a tracer with the @jit decorator.

# 3. Core Components of Avior

The building blocks of Avior are separated via a '**registry-based**' design for composability.
You can declare or register your providers and models, giving Avior information about rate-limits, costs, etc so it can optimize (more on this later). You can also declare or register new **Operators**, **Datasets**, **Evaluation Functions**, and **Prompts Signatures** in the `registry` directory.

## 3.1 Model Registry

The Model Registry centralizes all the model configurations and metadata. It's the backbone of Avior's approach to discovering and managing LLM endpoints (like **OpenAI**'s `gpt-4o`, **Anthropic**'s `claude-3.5-sonnet`, **Google**'s `gemini-1.5-pro`, proprietary and OSS models via **IBM WatsonX** or custom/OSS LLMs e.g. served via **Avior**'s serving endpoint's for OSS models like DBRX, Llama, Deepseek, Qwen, etc.)

### Key Features
- **Registration**: You define a ModelInfo object (containing metadata about the cost, rate limits, provider info, and an API key) and pass it to the registry.
- **Lookup**: You can retrieve models by strine ID or enum or easy referencing in your code. 
- **Usage Tracking**: Avior can track usage, cost, and latency logs for models and providers, and provide usage summaries.


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
resp = model_service("openai:gpt-4o", "Hello from Avior!")
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
- **Integrate easily with Avior’s concurrency model** (fan-out calls can just invoke multiple LM Modules in parallel).

### Basic Pattern

```python
from avior.registry.operator.operator_base import LMModuleConfig, LMModule

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
- **Chain-of-thought**: If you want to do a multi-step reasoning workflow, you can pass partial outputs into subsequent LM calls, all orchestrated by your Avior operator logic.


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
- **DSPy synergy**: In the future, Avior can seamlessly import DSPy “specifications” as Signatures.

# 5. Operators 

Operators are the 'workhorse' components of Avior. If you're familiar with PyTorch, think of them like `nn.Module`s.

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
If there's a signature, Avior checks if the input matches input_model or required_inputs. 

3. Execution:
- forward (inputs): runs synchronously in python, returns immediate results. 
- to_plan (inputs): returns an ExecutionPlan if you want Avior's scheduler to handle concurrency and optimize execution. 
 

5. (Optional) Output validation:
If there's a structured_output in the signature, Avior attempts to cast/validate the result.


## 5.2 Common Operator Types

- **RECURRENT**: Shape-preserving operators that refine or iterate on inputs over multiple steps. E.g., a self-refinement operator that calls a model repeatedly until convergence
- **FAN_IN**: Dimensionality-reducing operators that combine, merge, or select amonst multiple inputs (e.g., aggregator, judge, verifier-based judge, rank-filter, or majority-vote). 
- **FAN_OUT**: Dimensionality-expanding operators that produce multiple outputs from a single input (e.g., ensemble operators, or calls that split or replicate an input, such as prompt mixin operators that take in a single prompt and remix it by adding prefixes, suffixes, etc DSPy optimizer search style).

## 5.3 Example: JudgeExplainerOperator
Takes in multiple responses in a structured input, passes those calls to a judge lm call, returns two outputs: a final answer and an explanation in a structured output with a signature / prompt that asks for an explanation and a parse method call or GetAnswer op call or something to get afinite MCQ-compatible output, or True/False. 

```python
from collections import Counter
from avior.registry.operator.operator_base import Operator

class JudgeExplainerOperator(Operator):

```
### Why This Matters:
It’s **short**, **explicit**, and **composable**. You can feed multiple model outputs into this operator to get a final “sythesized” verdict.


# 6. GraphExecution and Scheduling

While operators can run eagerly, Avior also enables **graph-based** orchestration:
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
from avior.registry.model.model_registry import ModelRegistry
from avior.registry.operator.operator_registry import EnsembleOperator, GetAnswerOperator
from avior.registry.operator.operator_base import LMModuleConfig, LMModule

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
from avior.core.graph_executor import NoNGraphData, GraphExecutorService

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
input_data = {"query": "Explain how to set up Avior for multi-model parallel usage"}

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
	•	Avior automatically manages concurrency for the fan-out ensemble if you implement or use the operator’s to_plan() method.
	•	You can easily swap the aggregator with a “MostCommonOperator” or any custom logic.
	•	You can nest subgraphs or add more nodes as your pipeline grows in complexity.


# Next steps:

In the next sections, we'll cover installation, quickstart instructions, advancd topics (like DSPy-based signature integration and LiteLLM synergy), and details on contributing to Avior's ecosystem.


## Package Structure/'

```markdown
avior/
├── core/
│   ├── configs/
│   ├── non.py
├── registry/
│   ├── dataset_registry.py
│   ├── eval_function_registry.py
│   ├── model_registry.py
│   ├── operator_base.py
│   ├── operator_registry.py
│   ├── persona_registry.py
│   └── prompt_registry.py

```

## Key Components

### Operators

Operators are the fundamental building blocks of NoN systems in Avior. They encapsulate specific AI operations or transformations. Each operator is designed to be simple, explicit, and composable. The `OperatorBase` class serves as the foundation for all operators, analogous to PyTorch's `nn.Module`. This makes Avior structurally familiar to PyTorch users, enhancing approachability and ease of use.

#### Operator Types

- **RECURRENT**: Shape-preserving operators that refine or iterate on inputs over multiple steps.
- **FAN_IN**: Dimensionality-reducing operators that combine multiple inputs into one output.
- **FAN_OUT**: Dimensionality-expanding operators that produce multiple outputs from a single input (e.g., ensemble operators).

#### Example Operators

1. **EnsembleOperator (E)**
   
   Runs an ensemble of models to generate multiple responses.
   
   *Type: FAN_OUT.*
   
   ```python
   class EnsembleOperator(Operator):
       @classmethod
       def get_metadata(cls) -> OperatorMetadata:
           return OperatorMetadata(
               code='ENSEMBLE',
               description="Runs an ensemble of models to generate responses",
               operator_type=OperatorType.FAN_OUT
           )
   ```

2. **JudgeBasedOperator (JB)**
   
   Makes a judge-based decision based on advisor responses.
   
   *Type: FAN_IN.*

3. **MostCommonOperator (MC)**
   
   Determines the most common answer among multiple responses.
   
   *Type: FAN_IN.*

4. **SelfRefinementOperator (SR)**
   
   Performs iterative self-refinement on the given input/output.
   
   *Type: RECURRENT.*

### NoNGraph

The `NoNGraph` class is the core abstraction in Avior's compositional architecture. It allows you to define complex AI systems by connecting operators in a flexible, graph-like structure. This approach enables the creation of sophisticated AI pipelines while maintaining simplicity and explicitness in the design.

## Usage Examples

### Defining a Simple NoNGraph

```python
from avior.core.non import NoNGraph
from avior.registry.operator_registry import OperatorFactory

class ExampleNoN(NoNGraph):
    def __init__(self):
        super().__init__()
        
        # Define a basic graph structure with 3 gpt-4o models as ensemble members
        # and a single MostCommon operator to select the most common answer.
        self.add_node("node1", OperatorFactory.create('E', [{'model_name': 'gpt-4o', 'count': 3}]))
        self.add_node("node2", OperatorFactory.create('MC'), inputs=["node1"])
```

In this example, we create a `NoNGraph` with two nodes:

- An Ensemble operator (`E`) with three GPT-4o models.
- A MostCommon operator (`MC`) that takes inputs from the Ensemble operator and selects the most common answer.

This composition illustrates how you can build compound AI systems from simple, reusable components. This approach, leveraging multiple model instances and consensus-based decision making, aligns with insights from "More LLM Calls Are All You Need" (Chen et al., 2024), potentially improving robustness and accuracy in downstream tasks.

### Alternative Definition Using Shorthand

Avior offers a more concise string-based shorthand for defining graphs:

```python
def define_non_graph():
    # The shorthand "3:E:gpt-4o" means an Ensemble operator with 3 units of gpt-4o.
    # "1:MC" means a MostCommon operator with a single unit.
    graph_def = [
        ["3:E:gpt-4o"], 
        "1:MC"
    ]
    non_graph = NoNGraph().parse_from_list(graph_def)
    return non_graph
```

Both methods achieve the same result: instantiating a NoN that uses an ensemble of GPT-4o models and then selects a final answer using a MostCommon operator.

### Using Multiple Models in a NoN

You can easily mix different models and operators in a single `NoNGraph`:

```python
from avior.core.non import NoNGraph
from avior.registry.operator_registry import OperatorFactory

class MultiModelNoN(NoNGraph):
    def __init__(self):
        super().__init__()
        
        # node1: Ensemble of 3 gpt-4o models
        self.add_node("node1", OperatorFactory.create('E', [{'model_name': 'gpt-4o', 'count': 3}]))
        
        # node2: 1 claude-3.5-opus model
        self.add_node("node2", OperatorFactory.create('E', [{'model_name': 'claude-3.5-opus', 'count': 1}]))
        
        # node3: Ensemble of 2 gemini-1.5-pro models
        self.add_node("node3", OperatorFactory.create('E', [{'model_name': 'gemini-1.5-pro', 'count': 2}]))
        
        # node4: A MostCommon operator that aggregates outputs from node1, node2, and node3
        self.add_node("node4", OperatorFactory.create('MC', [{'model_name': 'gpt-4-turbo', 'count': 1}]),
                      inputs=["node1", "node2", "node3"])
```

#### Shorthand Version

```python
def define_multi_model_non_graph():
    # Here we have multiple ensembles followed by an MC operator:
    # "3:E:gpt-4o", "1:E:claude-3.5-opus", "2:E:gemini-1.5-pro", and then "1:MC:gpt-4-turbo".
    graph_def = [
        ["3:E:gpt-4o", "1:E:claude-3.5-opus", "2:E:gemini-1.5-pro"], 
        "1:MC:gpt-4-turbo"
    ]
    non_graph = NoNGraph().parse_from_list(graph_def)
    return non_graph
```

## Design Philosophy

Avior's API is inspired by PyTorch's principles, emphasizing:

1. **Composability**: Like `nn.Module`, Avior's Operators and NoNGraphs are designed to be easily composable, enabling complex AI systems from simple parts.

2. **Extensibility**: The registry-based approach allows effortless addition of new operators, models, datasets, evaluation functions, and prompts without changing core code.

3. **Usability**: Prioritizes explicitness and clarity over implicit, "magical" functionality. While this may require more explicit code, it makes the system easier to understand, debug, and maintain.

4. **Simple Over Easy**: Following PyTorch and the Zen of Python, Avior opts for explicit and understandable building blocks rather than highly abstracted or opinionated APIs.

### Avior’s Goals

- Provide a compositional, extensible framework for building NoN compound AI systems.
- Offer a structure familiar to researchers and developers with PyTorch experience.
- Enable rapid prototyping while maintaining explicitness and clarity.
- Not primarily focused (yet) on optimizing latency for massive future NoN systems, but rather on enabling research, experimentation, and development of new AI workflows.

## Extensibility

Avior is designed with extensibility in mind:

- **Datasets**: Add new datasets by creating custom parsers and registering them in `dataset_registry.py`.
- **Operators**: Create new operators by subclassing `OperatorBase` and registering them in `operator_registry.py`.
- **Prompts**: Add new prompts by registering them with `PromptRegistry`.
- **Evaluation Functions**: Implement custom evaluation functions and integrate them via `eval_function_registry.py`.
- **Models**: Add support for new model providers by implementing and registering them with `ModelRegistry`.

## Models

Avior supports multiple model providers and model types via `ModelRegistry`:

### Supported/Planned Model Types

- **OpenAI models** (e.g., gpt-4o, gpt-4-turbo, gpt-4o-mini)
- **Anthropic models** (e.g., claude-3.5-sonnet, claude-3-opus)
- **Google models** (e.g., gemini-1.5-pro, gemini-1.5-flash)
- **OSS models** (e.g., llama-3.1-8b, llama-3.1-70b, llama-3.1-405b) *[coming soon]*

### Example: Registering a Custom Model

```python
from avior.registry.model_registry import ModelRegistry, ModelInfo, ProviderInfo, ModelCost, RateLimit

custom_model_info = ModelInfo(
    model_id='custom-model',
    model_name='Custom Model',
    cost=ModelCost(input_cost_per_thousand=0.01, output_cost_per_thousand=0.02),
    rate_limit=RateLimit(tokens_per_minute=60000, requests_per_minute=3000),
    provider=ProviderInfo(name="CustomProvider", default_api_key="your_api_key"),
    api_key="your_api_key"
)

# CustomModelImplementation should be a class implementing the required model interface.
ModelRegistry.register_model(custom_model_info, CustomModelImplementation)
```

## Conclusion

Avior provides a compositional, extensible framework for building NoN compound AI systems. By following design principles akin to PyTorch, it offers a familiar and approachable structure for researchers and developers. Emphasizing explicit building blocks makes systems more understandable and manageable, even as they scale in complexity. With Avior, you can experiment with multiple models, operators, and consensus mechanisms, potentially improving the robustness and accuracy of AI-driven applications.

# Quick Start

**Clone the repository:**

```bash
git clone https://github.com/foundrytechnologies/avior.git
```

**Install the pre-requisite libraries:**

```bash
cd avior
pip install -e .
```

**Obtain an OpenAI API key and create a local config file (`config.ini`), replacing `MY_OPENAI_API_KEY` with a valid key:**

```bash
cat <<"EOF" >> ./config.ini
[models]
openai_api_key = MY_OPENAI_API_KEY
EOF
```

**Run an example (e.g., an MCQ experiment):**

```bash
python examples/mcq_experiement_example.py
```

## Contributing

Contributions are welcome! Whether it's adding new datasets, operators, prompts, models, or evaluation functions, your input helps improve the framework for everyone. See `CONTRIBUTING.md` for instructions on getting started.

## Quick Start Development

**Install extra requirements for development and testing:**

```bash
pip install -e .[dev]
```

**Run tests (from the avior directory):**

```bash
python -m unittest discover ./src/avior
```

See [Python unittest docs](https://docs.python.org/3/library/unittest.html) for additional options.

## License

Apache 2.0 License.


