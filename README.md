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

Ember's compact notation lets you express sophisticated AI architectures concisely:

```python
from ember.non import build_graph

# 5 parallel GPT-4o instances synthesized by Claude
system = build_graph(["5E@openai/gpt-4o(temp=0.7)", "1J@anthropic/claude-4-sonnet"])
result = system(query="What's the most effective approach to climate change?")

# With majority voting instead of judge synthesis
voting = build_graph(["7E@openai/gpt-4o(temp=0.7)", "1M"])

# Multi-branch architecture with verification
advanced = build_graph([
    "3E@openai/gpt-4o(temp=0.6)",      # Ensemble generation
    "1V@anthropic/claude-4-sonnet",     # Verification
    "1J@anthropic/claude-4-sonnet"      # Final synthesis
])
```

**Node types:** `E` (Ensemble), `J` (Judge), `M` (MostCommon), `V` (Verifier)

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

```python
from ember.api import operators, models

@operators.op
def summarize(text: str) -> str:
    return models("gpt-4o", f"Summarize: {text}")

@operators.op
def translate(text: str, lang: str = "Spanish") -> str:
    return models("gpt-4o", f"Translate to {lang}: {text}")

# Compose with >> operator
pipeline = summarize >> translate
result = pipeline("Long technical article...")
```

### 3. Data and Evaluation

```python
from pathlib import Path

from ember import data, models
from ember.eval import NumericAnswer, run

def solve(example: data.Example) -> str:
    return models("openai/gpt-5", example.question).text

records = run(
    solve,
    data.stream("gsm8k", limit=50),  # needs: pip install 'ember-ai[data]'
    scorer=NumericAnswer(),
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
