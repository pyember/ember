# Ember Integrations

This directory contains official integrations for Ember with other AI frameworks and tools.

**Important**: These integrations are NOT included in the base `pip install ember-ai` package. They may be installed separately or included as extras.

## Available Integrations

### 1. DSPy Integration
Enables using Ember models within DSPy programs.

```bash
# Install (future)
pip install ember-ai[dspy]

# Usage
from ember.integrations.dspy import EmberLM
lm = EmberLM(model="gpt-4")
```

### 2. Swarm Integration  
Multi-agent orchestration using OpenAI's Swarm pattern.

```bash
# Install (future)
pip install ember-ai[swarm]

# Usage
from ember.integrations.swarm import EmberSwarmClient
client = EmberSwarmClient()
```

### 3. MCP (Model Context Protocol) Integration
Expose Ember models and graph tooling via the Model Context Protocol.

```bash
# Install (future)
pip install ember-ai[mcp]

# Usage
from ember.integrations.mcp import EmberMCPServer
server = EmberMCPServer()
```

**Graph Advisor tools (beta):**
- `ember_graph_advisor.plan` – validates a compact NON spec and returns the normalized graph plus usage estimate.
- `ember_graph_advisor.run` – executes the graph (candidates → judge) and streams `candidate`, `judge`, `status`, and `complete` events.

**Configuration essentials:**
- Enable with `mcp.graph_advisor.enabled: true` in `~/.ember/config.yaml`.
- Register allowed MCP clients and per-tool rate limits under `mcp.graph_advisor.allowed_clients` and `mcp.graph_advisor.rate_limits`.
- Ensure model bindings referenced in graph specs are present in the MCP `ModelToolRegistry`.

## Installation Options

### Option 1: Install Specific Integration (Future)
```bash
pip install ember-ai[dspy]
pip install ember-ai[swarm]
pip install ember-ai[mcp]
```

### Option 2: Install All Integrations (Future)
```bash
pip install ember-ai[all]
```

### Option 3: Development Installation (Current)
```bash
# Clone the repository
git clone https://github.com/anthropics/ember
cd ember

# Install in development mode
pip install -e .

# Integrations are now available
from ember.integrations.dspy import EmberLM
```

## Architecture Philosophy

Each integration:
- Is self-contained with minimal dependencies
- Follows the adapter pattern
- Preserves the semantics of the integrated framework
- Adds Ember's benefits (unified API, monitoring, optimization)

## Observability

Integrations share a lightweight ``MetricsContext`` (``from ember._internal.context import MetricsContext``)
that records latency, success/failure, and token usage for every model call. The
context exposes ``track()`` spans that wrap model invocations and surface both
the most recent measurements (``get_last_metrics()``) and rolling aggregates
(``get_aggregated_metrics()``) so SDKs can emit diagnostics without depending on
gateway services.

## Creating New Integrations

To add a new integration:

1. Create a new directory: `integrations/your_framework/`
2. Implement the adapter pattern
3. Include examples and tests
4. Update this README
5. Submit a PR

### Integration Template
```python
# integrations/your_framework/__init__.py
from ember.models import ModelRegistry

class YourFrameworkAdapter:
    def __init__(self, model_id: str):
        self.ember_model = ModelRegistry().get_model(model_id)
        
    def framework_specific_method(self, *args, **kwargs):
        # Adapt framework calls to Ember
        response = self.ember_model.generate(...)
        # Adapt Ember response to framework format
        return framework_response
```

## Future Roadmap

- **LangChain Integration**: Complete LangChain compatibility
- **LlamaIndex Integration**: Document stores and retrievers
- **Haystack Integration**: Pipeline components
- **Weights & Biases Integration**: Experiment tracking
- **MLflow Integration**: Model versioning

## Contributing

Integrations are a great way to contribute to Ember! See our [Contributing Guide](../CONTRIBUTING.md) for details.
