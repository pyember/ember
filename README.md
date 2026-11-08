<p align="center">
  <img src="docs/assets/logo_ember_icon@2x.png" alt="Ember Logo" width="120"/>
</p>
<p align="center">
  <img src="docs/assets/ember_workmark.svg" alt="Ember" width="280"/>
</p>

<p align="center">
  <strong>Build AI systems with the elegance of print("Hello World")</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/ember-ai/"><img src="https://img.shields.io/pypi/v/ember-ai.svg" alt="PyPI"></a>
</p>

---

Ember is a compositional framework for compound AI systems. Think of it as PyTorch for "Networks of Networks" (NONs)—where neural networks compose neurons, Ember composes LLM calls into sophisticated reasoning architectures with automatic parallelization.

## Quick Start

```python
from ember.api import models

# One line to call any model
response = models("gpt-4o", "Explain quantum computing")
print(response)
```

### Installation

```bash
pip install ember-ai
```

Or from source:

```bash
git clone https://github.com/pyember/ember.git
cd ember
uv sync
```

### Setup

```bash
export OPENAI_API_KEY=...          # or: ember configure set-key openai
```

## NON: Compound AI in One Line

A NON specification names a network of model calls in one line, and `build_graph` turns it into an
operator that runs:

```python
from ember import build_graph

# Five samples, then a judge that reads all five and writes the answer.
graph = build_graph("5E@openai/gpt-5(temperature=0.7), 1J@anthropic/claude-sonnet-4-5")
answer = graph("What is the most effective way to cut household energy use?")
print(answer.text)

# Seven samples and a majority vote.
voted = build_graph("7E@openai/gpt-5(temperature=0.7), 1M")
print(voted("Is 1009 prime? Answer yes or no.").text)

# Samples from two models, a verifier that keeps the answers it passes, then a judge.
checked = build_graph(
    "3E@openai/gpt-5, 3E@anthropic/claude-sonnet-4-5, 1V@openai/gpt-5, "
    "1J@anthropic/claude-sonnet-4-5"
)
print(checked("Why is the sky blue? Two sentences.").text)
```

| Stage | What it does |
|-------|--------------|
| `nE@model` | `n` calls to the model with the question. All E stages run concurrently. |
| `nJ@model` | `n` judges, each reading the question and every answer; their answers replace the answers. |
| `1V@model` | Checks each answer with its own call and keeps the ones that pass. |
| `1M` | Majority vote: the answer most calls agree on, compared case- and space-insensitively. |

A graph starts with E stages and ends with `1J` or `1M`. Model ids are `provider/name` and are passed
through untouched, so `ollama/llama3.1:8b` works. E, J and V take `temperature`, `top_p`,
`max_output_tokens`, `seed`, `reasoning_effort`, `system` and `quorum`, the number of calls that must
succeed (for V, answers that must pass). A mistake raises `NonSyntaxError` pointing at its column.
`answer.stages` keeps every call's outcome, including failures.

## Core Concepts

Ember provides four primitives that compose into powerful AI systems:

### 1. Models — Direct LLM Access

```python
from ember.api import models

# Simple calls with any provider
response = models("gpt-4o", "Write a haiku")
response = models("claude-4-sonnet", "Explain recursion")
response = models("gemini-2.5-pro", "Summarize this text")

# Reusable configured instance
assistant = models.instance("gpt-4o", temperature=0.7, system="You are helpful")
response = assistant("How do I center a div?")
```

### 2. Operators — Composable Building Blocks

An operator is a frozen dataclass with a `forward` method. Functions become operators with `@op`,
and `>>` chains them, with mypy checking that each stage takes what the previous one returns:

```python
import dataclasses

from ember import op
from ember.operators import Ensemble, ModelCall, Operator, Succeeded


@op
def tidy(text: str) -> str:
    return " ".join(text.split())


summarize = ModelCall("openai/gpt-5", template="Summarize in one sentence:\n{input}")
pipeline = tidy >> summarize
print(pipeline("  Ember composes   model calls.  ").text)


@dataclasses.dataclass(frozen=True, slots=True)
class Exclaim(Operator[str, str]):
    mark: str = "!"

    def forward(self, input: str) -> str:
        return input + self.mark


louder = dataclasses.replace(Exclaim(), mark="!!!")  # Operators never change; replace copies.

# Members run concurrently; every member's outcome comes back, failures included.
panel = Ensemble(
    (ModelCall("openai/gpt-5"), ModelCall("anthropic/claude-sonnet-4-5")),
    quorum=1,
)
for outcome in panel("Name a prime number."):
    if isinstance(outcome, Succeeded):
        print(outcome.member, outcome.value.text)
    else:
        print(outcome.member, "failed:", outcome.error)
```

An ensemble records rate limits, overload, server and transport errors as failed members, and raises
`QuorumError` when fewer than `quorum` members succeed. Errors that would fail every member alike,
such as a bad key or an unknown model, are raised at once. `Judge`, `MajorityVote` and `Verifier`
take a question with its candidate answers.

Validation is optional and declared once, as pydantic models on the class:

```python
import dataclasses

import pydantic

from ember.operators import Operator


class Question(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class Length(Operator[Question, int]):
    input_model = Question  # Checked strictly on every call.

    def forward(self, input: Question) -> int:
        return len(input.text)


print(Length()({"text": "hello"}))  # 5; {"text": 5} or an extra key raises ValidationError
```

### 3. Data and Evaluation

```python
from pathlib import Path

from ember import data, models
from ember.eval import NumericAnswer, run

def solve(example: data.Example) -> str:
    prompt = f"{example.question}\n\nEnd with a line of the form 'Answer: <number>'."
    return models("openai/gpt-5", prompt).text

records = run(
    solve,
    data.stream("gsm8k", limit=50),  # needs: pip install 'ember-ai[data]'
    scorer=NumericAnswer(after="Answer:"),  # reads the number after the last "Answer:"
    records=Path("gsm8k.jsonl"),  # every item and every request attempt
)
print(records.summary)  # score, failures, retries, unpriced calls, cost
```

### 4. XCS — Automatic Optimization

```python
from ember import xcs

@xcs.jit  # Automatic parallelization
def analyze_batch(items):
    return [models("gpt-4o", item) for item in items]

# vmap for vectorized execution
fast_analyze = xcs.vmap(analyze_batch)
results = fast_analyze(large_dataset)
```

## Local Models (No API Keys)

Try Ember with Ollama—no cloud accounts needed:

```bash
# Install Ollama
brew install ollama && ollama serve  # macOS
# Pull a model
ollama pull llama3.2:1b
```

```python
from ember.api import models
print(models("ollama/llama3.2:1b", "Hello from Ember!"))
```

## Available Models

Ember auto-discovers models from your configured providers:

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-5, gpt-4.1, gpt-4o, gpt-4o-mini, o1 |
| **Anthropic** | claude-4-sonnet, claude-opus-4, claude-3.5-sonnet-latest |
| **Google** | gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro-latest |

## Configuration

Ember reads keys from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`, or from
`~/.ember/config.yaml` (or the file named by `EMBER_CONFIG`), which the CLI writes with mode 0600:

```bash
ember configure set-key openai      # prompts for the key; keys are never command-line arguments
ember configure set local kind openai_compatible
ember configure set local base_url http://localhost:11434/v1
ember configure show                # keys are masked
ember check gpt-5.5                 # one short request: finish reason, latency, request id, cost
```

Runtime overrides apply to a block and are never written to disk:

```python
from ember import context

with context.override(providers={"openai": {"api_key": "sk-..."}}):
    response = models("openai/gpt-5", "Hello!")
```

## CLI Reference

```bash
ember configure show                          # print the config file, keys masked
ember configure set-key PROVIDER              # save a key typed at a hidden prompt
ember configure set PROVIDER FIELD VALUE      # FIELD is kind, base_url or api_key_env
ember configure unset PROVIDER [FIELD]        # remove a field (even a misspelt one) or a provider
ember models                                  # list the model catalog with limits and prices
ember check MODEL                             # send one short request and report the result
```

## Design Principles

1. **Simple by Default** — Basic usage requires no configuration
2. **Progressive Disclosure** — Complexity available when needed
3. **Composition Over Configuration** — Build complex from simple
4. **Automatic Optimization** — Fast by default, no manual tuning

## Architecture

- **Model Registry** — Multi-provider LLM management with adaptive rate limiting
- **Operator System** — Composable units with JAX-style transforms
- **NON Engine** — Compact notation for compound AI architectures
- **XCS Runtime** — Automatic parallelization and optimization

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Development

```bash
git clone https://github.com/pyember/ember.git
cd ember
uv sync --all-extras

uv run pytest              # Run tests
uv run mypy src/           # Type checking
uv run ruff check . --fix  # Linting
```

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE).

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full list of contributors from Foundry, Stanford, UC Berkeley, IBM Research, Databricks, and other institutions who helped shape Ember.
