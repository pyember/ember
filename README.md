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
ember setup  # Interactive wizard configures API keys
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

### 3. Data — Streaming Pipelines

```python
from ember.api import data, models

# Stream and process datasets efficiently
for example in data.stream("mmlu"):
    answer = models("gpt-4o", example["question"])
    print(f"Q: {example['question']}\nA: {answer}\n")

# Chain transformations
results = (data.stream("gsm8k")
    .filter(lambda x: x["difficulty"] > 7)
    .transform(preprocess)
    .batch(32))
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

```bash
# Interactive setup (recommended)
ember setup

# Manual configuration
ember configure set providers.openai.api_key "sk-..."
ember configure set providers.anthropic.api_key "sk-ant-..."

# Verify
ember test
```

Runtime overrides:

```python
from ember.context import context

with context.manager(providers={"openai": {"api_key": "sk-..."}}):
    response = models("gpt-4o", "Hello!")
```

## CLI Reference

```bash
ember setup                    # Interactive setup wizard
ember test                     # Test API connections
ember models                   # List available models
ember configure list           # Show configuration
ember context view             # View runtime context
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
