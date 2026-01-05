# Ember

Build AI systems with the elegance of print("Hello World").

## Installation

### From PyPI

```bash
pip install ember-ai
```

### From Source (Development)

```bash
git clone https://github.com/pyember/ember.git
cd ember
uv sync
```

## Quick Setup

Run our interactive setup wizard for the best experience:

```bash
# If installed from PyPI
ember setup

# If running from source
uv run ember setup
```

This will:
- Help you choose an AI provider (OpenAI, Anthropic, or Google)
- Configure your API keys securely  
- Test your connection
- Save configuration to ~/.ember/config.yaml

## Getting Started

### 1. Configure API Keys

Ember now reads provider credentials exclusively from its configuration file.
Use the CLI to persist secrets:

```bash
# Interactive wizard (writes to ~/.ember/config.yaml)
uv run ember setup

# Or set values directly
uv run ember configure set providers.openai.api_key "sk-..."
uv run ember configure set providers.anthropic.api_key "sk-ant-..."
uv run ember configure set providers.google.api_key "..."
```

Environment-variable fallbacks (for example `OPENAI_API_KEY`)
are deprecated and no longer read at runtime. Migrate any existing shell scripts
or CI jobs to use `ember configure` so credentials remain deterministic.

**Option 3: Runtime Context**
```python
from ember.api import models
from ember.context import context

with context.manager(providers={"openai": {"api_key": "sk-..."}}):
    response = models("gpt-4", "Hello!")
```

### 2. Verify Setup

```python
from ember.api import models

# Discover available models
print(models.list())  # Shows all available models
print(models.providers())  # Shows available providers

# Get detailed model information
info = models.discover()
for model_id, details in info.items():
    print(f"{model_id}: {details['description']} (context: {details['context_window']})")

# This will work once credentials are configured via `ember setup` / `ember configure`
response = models("gpt-4", "Hello, world!")
print(response)
```

If credentials are missing, you'll get a clear error message:
```
ModelProviderError: No API key found for gpt-4.

To fix this, choose one:

Option 1: Run interactive setup (recommended)
   ember setup

Option 2: Save to config
   ember configure set providers.openai.api_key YOUR_KEY

Get your API key from: https://platform.openai.com/api-keys
```

If you use an unknown model name, you'll see available options:
```
ModelNotFoundError: Cannot determine provider for model 'claude-3'. 
Available models: claude-4-sonnet, claude-3.5-sonnet-latest, claude-3.5-haiku-latest, 
gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro-latest, gpt-5, gpt-4.1, gpt-4o, 
gpt-4o-mini, ...
```

### 3. Choose Your Style: Strings or Constants

```python
from ember.api import models, Models

# Option 1: Direct strings (simple, works everywhere)
response = models("gpt-4", "Hello, world!")
response = models("claude-4-sonnet", "Write a haiku")

# Option 2: Constants for IDE autocomplete and typo prevention
response = models(Models.GPT_4, "Hello, world!")
response = models(Models.CLAUDE_3_OPUS, "Write a haiku")

# Both are exactly equivalent - Models.GPT_4 == "gpt-4"
```

## Quick Start

```python
from ember.api import models

# Direct LLM calls - no setup required
response = models("gpt-4", "Explain quantum computing in one sentence")
print(response)
```

### Local, No API Keys (Ollama)

Try Ember without creating any cloud accounts by using a local model via Ollama:

1) Install and run Ollama

- macOS: `brew install ollama && ollama serve`
- Linux: `curl -fsSL https://ollama.com/install.sh | sh && ollama serve`

2) Pull a model the first time (fast option)

```
ollama run llama3.2:1b
```

3) Call from Ember (auto-pull supported)

```python
from ember.api import models
print(models("ollama/llama3.2:1b", "Say hi from Ember", autopull=True))
```

Troubleshooting
- Connection refused: ensure `ollama serve` is running, or set `OLLAMA_BASE_URL` if using a non-default host/port.
- Model not found: run `ollama run <model>` once to download it locally.
- Slow/timeout: increase timeout via `EMBER_OLLAMA_TIMEOUT_MS` or pass `timeout=60` in the call.
- Streaming: `stream=True` is aggregated in this version; streaming iterator support is planned.

### Available Models

Common model identifiers:
- **OpenAI**: `gpt-5`, `gpt-4.1`, `gpt-4o`, `o1`, `gpt-4o-mini`
- **Anthropic**: `claude-4-sonnet`, `claude-opus-4`, `claude-3.5-sonnet-latest`
- **Google**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`

Discovery runs by default, so as soon as your API key can see a newly released model (for example `gemini-2.5-flash` or `claude-4-sonnet`), Ember learns about it without code changes.

Models are automatically routed to the correct provider based on their name.

## Core Patterns

### Context Management

Ember uses a unified context system for configuration and state management:

```python
from ember.context import context

# Get the current context
ctx = context.get()

# Temporary configuration overrides
with context.manager(models={"default": "gpt-4", "temperature": 0.9}) as ctx:
    # All operations in this block use these settings
    response = models("Hello")  # Uses gpt-4 with temperature 0.9
```

The context system provides:
- Thread-safe and async-safe configuration management
- Hierarchical configuration with proper isolation
- Clean scoping for temporary overrides

### Progressive Disclosure in APIs

Ember APIs follow a pattern of progressive disclosure - simple things are simple, complex things are possible:

```python
from ember.api import models

# Level 1: Simple one-off calls
response = models("gpt-4", "Hello world")

# Level 2: Reusable configured instances  
assistant = models.instance("gpt-4", temperature=0.7, system="You are helpful")
response = assistant("How do I center a div?")
```

This pattern appears throughout Ember:
- `models()` for quick calls, `models.instance()` for configured instances
- `operators.op` for simple functions, full `Operator` classes for complex needs
- Direct data access for simple cases, streaming pipelines for scale

## Core Concepts

Ember provides four primitives that compose into powerful AI systems:

### 1. Models - Direct LLM Access

```python
from ember.api import models

# Simple invocation with string model names
response = models("claude-4-sonnet", "Write a haiku about programming")

# Reusable configuration
assistant = models.instance("gpt-4", temperature=0.7, system="You are helpful")
response = assistant("How do I center a div?")

# Alternative: Use constants for autocomplete
from ember.api import Models
response = models(Models.GPT_4, "Write a haiku")
```

### 2. Operators - Composable AI Building Blocks

```python
from ember.api import operators

# Transform any function into an AI operator
@operators.op
def summarize(text: str) -> str:
    return models("gpt-4", f"Summarize in one sentence: {text}")

@operators.op
def translate(text: str, language: str = "Spanish") -> str:
    return models("gpt-4", f"Translate to {language}: {text}")

# Compose operators naturally
pipeline = summarize >> translate
result = pipeline("Long technical article...")
```

### 3. Data - Streaming-First Data Pipeline

```python
from ember.api import data

# Stream data efficiently
for example in data.stream("mmlu"):
    answer = models("gpt-4", example["question"])
    print(f"Q: {example['question']}")
    print(f"A: {answer}")

# Chain transformations
results = (data.stream("gsm8k")
    .filter(lambda x: x["difficulty"] > 7)
    .transform(preprocess)
    .batch(32))
```

### 4. XCS - Zero-Config Optimization

```python
from ember import xcs

# Automatic JIT compilation
@xcs.jit
def process_batch(items):
    return [models("gpt-4", item) for item in items]

# Automatic parallelization
fast_process = xcs.vmap(process_batch)
results = fast_process(large_dataset)  # Runs in parallel
```

## Real-World Examples

### Building a Code Reviewer

```python
from ember.api import models, operators

@operators.op
def review_code(code: str) -> dict:
    """AI-powered code review"""
    prompt = f"""Review this code for:
    1. Bugs and errors
    2. Performance issues
    3. Best practices
    
    Code:
    {code}
    """

    review = models("claude-4-sonnet", prompt)
    
    # Extract structured feedback
    return {
        "summary": models("gpt-4", f"Summarize in one line: {review}"),
        "issues": review,
        "severity": models("gpt-4", f"Rate severity 1-10: {review}")
    }

# Use directly
feedback = review_code("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""")
```

### Parallel Document Processing

```python
from ember import xcs
from ember.api import data, models

# Define processing pipeline
@xcs.jit
def analyze_document(doc: dict) -> dict:
    # Extract key information
    summary = models("gpt-4", f"Summarize: {doc['content']}")
    entities = models("gpt-4", f"Extract entities: {doc['content']}")
    sentiment = models("gpt-4", f"Analyze sentiment: {doc['content']}")
    
    return {
        "id": doc["id"],
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment
    }

# Process documents in parallel
documents = data.stream("research_papers").first(1000)
results = xcs.vmap(analyze_document)(documents)

# Results computed optimally with automatic parallelization
```

### Multi-Model Ensemble

```python
from ember.api import models, operators

@operators.op
def consensus_answer(question: str) -> str:
    """Get consensus from multiple models"""
    # Query different models
    gpt4_answer = models("gpt-4", question)
    claude_answer = models("claude-4-sonnet", question)
    gemini_answer = models("gemini-1.5-pro-latest", question)
    
    # Synthesize consensus
    synthesis_prompt = f"""
    Question: {question}
    
    Model answers:
    - GPT-4: {gpt4_answer}
    - Claude: {claude_answer}
    - Gemini: {gemini_answer}
    
    Synthesize the best answer combining insights from all models.
    """
    
    return models("gpt-4", synthesis_prompt)

# Use for critical decisions
answer = consensus_answer("What's the best approach to distributed systems?")
```

## Command Line Interface

Ember provides a comprehensive CLI for setup, configuration, and introspection.

**Note:** If you installed Ember from PyPI, use `ember` directly. If running from source, prefix commands with `uv run`.

### Setup and Configuration

```bash
# Interactive setup wizard (recommended for first-time setup)
ember setup   # or: uv run ember setup

# Test your API connection
ember test   # or: uv run ember test
ember test --model claude-4-sonnet

# Configuration management
ember configure get models.default              # Get a config value
ember configure set models.default "gpt-4"     # Set a config value
ember configure list                            # Show all configuration
ember configure show credentials               # Show specific section

# Version information
ember version   # or: uv run ember version
```

### Introspection Commands

```bash
# Context introspection
ember context view                              # View current configuration
ember context view --format json                # Output as JSON
ember context view --filter models              # Show only models config
ember context validate                          # Validate configuration

# Registry introspection  
ember registry list-models                      # List available models
ember registry list-models --provider openai    # Filter by provider
ember registry list-models --verbose            # Detailed information
ember registry list-providers                   # Show provider status
ember registry info gpt-4                       # Detailed model info
```

### Advanced Configuration

The context system supports multiple configuration sources with priority:

1. **Runtime context** (highest priority)
2. **Configuration file** (`~/.ember/config.yaml`) + credential store
3. **Defaults** (lowest priority)

Environment variables are supported for non-sensitive toggles (and to point at a
config file via `EMBER_CONFIG_PATH`), but provider credentials are read from the
centralized config/credential store.

```python
from ember.api import models
from ember.context import context

# Use context manager for temporary overrides
with context.manager(
    models={"default": "gpt-4", "temperature": 0.7},
):
    # Production operations here
    response = models("gpt-4", "Production query")
```

## Advanced Features

### Type-Safe Operators

```python
from ember.api.operators import Operator
from pydantic import BaseModel

class CodeInput(BaseModel):
    language: str
    code: str

class CodeOutput(BaseModel):
    is_valid: bool
    errors: list[str]
    suggestions: list[str]

class CodeValidator(Operator):
    input_spec = CodeInput
    output_spec = CodeOutput
    
    def call(self, input: CodeInput) -> CodeOutput:
        prompt = f"Validate this {input.language} code: {input.code}"
        result = models("gpt-4", prompt)
        # Automatic validation against output_spec
        return CodeOutput(...)
```

### Custom Data Loaders

```python
from ember.api import data

# Register custom dataset
@data.register("my-dataset")
def load_my_data():
    with open("data.jsonl") as f:
        for line in f:
            yield json.loads(line)

# Use like built-in datasets
for item in data.stream("my-dataset"):
    process(item)
```

### Performance Profiling

```python
from ember import xcs

# Automatic profiling
with xcs.profile() as prof:
    results = expensive_operation()

print(prof.report())
# Shows execution time, parallelism achieved, bottlenecks
```

## Design Principles

1. **Simple by Default** - Basic usage requires no configuration
2. **Progressive Disclosure** - Complexity available when needed
3. **Composition Over Configuration** - Build complex from simple
4. **Performance Without Sacrifice** - Fast by default, no manual tuning

## Architecture

Ember uses a registry-based architecture with four main components:

- **Model Registry** - Manages LLM providers and connections
- **Operator System** - Composable computation units with JAX integration
- **Data Pipeline** - Streaming-first data loading and transformation
- **XCS Engine** - Automatic optimization and parallelization

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Development

```bash
# Clone and install development dependencies
git clone https://github.com/pyember/ember.git
cd ember
uv sync --all-extras

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Benchmarks
uv run python -m benchmarks.suite
```

## Contributing

We welcome contributions that align with Ember's philosophy of simplicity and power. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Ember is inspired by the engineering excellence of:
- JAX's functional transformations
- PyTorch's intuitive API
- Langchain's comprehensive features (but simpler)
- The Unix philosophy of composable tools

Built with principles from leaders who shaped modern computing.
