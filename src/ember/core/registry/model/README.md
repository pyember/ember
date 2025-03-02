# models Module (a.k.a. registry.model)

## Overview

The `models` module is responsible for:

1. **Model Discovery & Registration**
    *   We have a thread-safe registry (`ModelRegistry`) that stores `ModelInfo` objects and the corresponding provider model instances.
    *   A factory mechanism (`ModelFactory`) that creates provider-specific classes (e.g., `OpenAIModel`, `AnthropicModel`, etc.) from `ModelInfo`.
2. **Model Invocation**
    *   A high-level `ModelService` that looks up a model by ID or enum, calls it with a prompt/context, and optionally logs usage.
3. **Configuration**
    *   A `config.yaml` (and optionally `.env`) define model settings, such as cost, rate limits, base URLs, and API keys.
    *   The `config.py` file uses `pydantic-settings` to load environment variables and parse the YAML file for registry models.
4. **Usage Tracking**
    *   A `UsageService` that aggregates token counts, cost calculations, or any usage stats returned by each provider.

Overall, this module provides a "network-of-networks" building block for large-scale model usage in a "PyTorch / JAX-inspired" framework.

> **For a comprehensive quickstart guide, see [docs/quickstart/model_registry.md](/docs/quickstart/model_registry.md)**

## Package Layout

```
registry/
  model/
    ├── config.py               <-- Pydantic-based global emberSettings
    ├── config.yaml             <-- YAML-based model configs & defaults
    ├── provider_registry/      <-- Provider-specific model classes
    │    ├── base.py
    │    ├── openai_provider.py
    │    ├── anthropic_provider.py
    │    ├── gemini_provider.py
    │    └── ...
    ├── schemas/                <-- Pydantic data schemas (ModelInfo, Cost, RateLimit, etc.)
    │    ├── model_info.py
    │    ├── provider_info.py
    │    ├── cost.py
    │    ├── chat_schemas.py
    │    └── ...
    ├── services/               <-- High-level orchestration classes
    │    ├── model_service.py
    │    ├── usage_service.py
    │    └── ...
    ├── registry/               <-- Core registry + factory
    │    ├── model_registry.py
    │    ├── factory.py
    │    ├── model_enum.py
    │    └── ...
    ├── modules/                <-- Example LM modules (PyTorch-inspired usage)
    │    └── lm_modules.py
    └── ...
```

## Quick Start

1. **Install Dependencies**

   ```bash
   poetry install
   ```

2. **Set Environment Variables**  
   The easiest way is to create a `.env` file at the project root (same directory as `config.yaml` or wherever your Python code runs). For example:

   ```bash
   # .env
   OPENAI_API_KEY=sk-1234abcd
   ANTHROPIC_API_KEY=claude-xyz789
   GOOGLE_API_KEY=AIza-abc123
   ```

   Alternatively, you can export them in your shell/session:
   ```bash
   export OPENAI_API_KEY="sk-1234abcd"
   export ANTHROPIC_API_KEY="claude-xyz789"
   export GOOGLE_API_KEY="AIza-abc123"
   ```
   Note: These variables will be read automatically by pydantic-settings in `config.py`.

3. **Edit config.yaml**  
   In the `registry/model/config.yaml`, you can define top-level settings like `debug`, `logging`, and the `registry` section:

   ```yaml
   debug: true
   logging:
     level: "DEBUG"

   registry:
     auto_register: true
     # If you have other provider YAMLs, reference them here:
     included_configs:
       - "./src/ember/registry/model/provider_registry/openai_provider/openai_config.yaml"
       - "./src/ember/registry/model/provider_registry/anthropic_provider/anthropic_config.yaml"
       - "./src/ember/registry/model/provider_registry/deepmind/gemini_config.yaml"

     # Or inline your models directly:
     models:
       - model_id: "openai:gpt-4o"
         model_name: "GPT-4o"
         cost:
           input_cost_per_million: 5000
           output_cost_per_million: 15000
         rate_limit:
           tokens_per_minute: 10000000
           requests_per_minute: 1500
         provider:
           name: "OpenAI"
           default_api_key: "${OPENAI_API_KEY}"
           base_url: "https://api.openai.com"
         api_key: null
       ...
   ```
   • If `auto_register: true`, `initialize_global_registry()` will automatically register these models in the global registry.  
   • The syntax `"${OPENAI_API_KEY}"` tells the code to read from your environment variable.

4. **Run the Initialization & Use the Models**

   In your Python code (e.g., `main.py`):
   ```python
   from src.ember.registry.model.config import initialize_global_registry, GLOBAL_MODEL_REGISTRY
   from src.ember.registry.model.base.services.model_service import ModelService

   def main():
       # 1) Initialize the global registry
       initialize_global_registry()
       #    This reads config.yaml, merges any included provider configs,
       #    and registers all models.

       # 2) Create a ModelService that uses the global registry by default
       svc = ModelService()

       # 3) Invoke a model
       response = svc(prompt="Hello from GPT-4o!", model_id="openai:gpt-4o")
       print(response.data)

   if __name__ == "__main__":
       main()
   ```
   That's it! The `ModelService` will look up `"openai:gpt-4o"` in the `GLOBAL_MODEL_REGISTRY`, which was populated by `initialize_global_registry()`.

## Quick Start – One-Line Initialization

```python
import ember

# Automatically initialize registry and model service.
service = ember.init()
response = service("openai:gpt-4o", "Hello world!")
print(response.data)
```

## Usage Example

Below is a simplified usage example (see `example.py` in the `examples/` folder for a more detailed version):

```python
from src.ember.registry.model.base.services.model_service import ModelService
....
svc = ModelService()
response = svc(model_id="openai:gpt-4o", prompt="Hello world!")
print("Response:", response.data)
```

You can also retrieve the model directly:

```python
model = svc.get_model("openai:gpt-4o")
response = model("What's the capital of France?")
print("Direct usage:", response.data)
```

## Environment Variables & pydantic-settings

• We use pydantic-settings in `config.py`, so it automatically loads environment variables or `.env` keys matching the fields defined in `emberSettings`.  
• Keys recognized by default:  
  - `OPENAI_API_KEY`  
  - `ANTHROPIC_API_KEY`  
  - `GOOGLE_API_KEY`  
• You can add more if you create new providers or rename existing environment-variable fields.

Pro Tip: If you want to store your `.env` file in a different path or name, you can configure that by specifying the `env_file` within `emberSettings` or via environment variables themselves.

## The config.py Flow

1. **Global Singletons**  
   `GLOBAL_MODEL_REGISTRY` and `GLOBAL_USAGE_SERVICE` are defined as module-level singletons.

2. **initialize_global_registry()**  
   • Reads your main `config.yaml` from `settings.model_config_path` (defaults to `"config.yaml"`).  
   • Looks for `registry.included_configs` (a list of additional `.yaml` files) and deep-merges them into the main config.  
   • If `registry.auto_register` is true, it iterates over the final `registry.models` list and registers each model in `GLOBAL_MODEL_REGISTRY`.  
   • Only runs once due to an internal `_INITIALIZED` flag (so you can safely call it multiple times without duplicating).

## Example Provider YAML (Local Config)

If you store additional models in a separate YAML file like `gemini_config.yaml`:

```yaml
models:
  - model_id: "google:gemini-1.5-pro"
    model_name: "Gemini 1.5 Pro"
    cost:
      input_cost_per_million: 3500
      output_cost_per_million: 10500
    rate_limit:
      tokens_per_minute: 1000000
      requests_per_minute: 1000
    provider:
      name: "Google"
      default_api_key: "${GOOGLE_API_KEY}"
      base_url: "https://api.google.com"
    api_key: null
  ...
```

Then reference it in `config.yaml` under `registry.included_configs`, and it will merge automatically.

## Adding a New Provider or Model

1. Create a subclass of `BaseProviderModel` in `provider_registry/your_provider.py`.
2. Add the provider name as an enum if desired, or just rely on the config's `provider.name`.
3. Add a new entry in `config.yaml`:

    ```yaml
    debug: true
    logging:
      level: "DEBUG"
    registry:
      auto_register: true
      models:
        - model_id: "myprovider:cool-model"
          model_name: "Cool Model"
          cost:
            input_cost_per_million: 200
            output_cost_per_million: 600
          rate_limit:
            tokens_per_minute: 500000
            requests_per_minute: 1000
          provider:
            name: "MyProvider"
            default_api_key: "${MYPROVIDER_API_KEY}"
          api_key: null
    ```

4. Run `example.py`. If the provider class is discovered via your scanning logic (or manually mapped in `factory.py`), it'll be accessible.

## Further Notes

*   **Thread Safety:** `ModelRegistry` uses a lock around model registration and retrieval to handle concurrent usage in multi-threaded scenarios.
*   **Usage Logging:** The `UsageService` is optional but recommended. If your provider returns token usage or cost data, you can store it in memory or persist it.
*   **Error Handling:** Custom exceptions (`ProviderConfigError`, etc.) provide explicit error messaging for misconfigurations.

## Simple "Happy Path" Usage

If you don't need advanced discovery or usage tracking, here's a minimal workflow:

1. **Create or update your config YAML** so it has your model's info (API key, cost, rate limits, etc.).  
2. **Initialize the global registry** in your application start-up code:

```python
from src.ember.core.registry.model.settings import initialize_global_registry

initialize_global_registry()  # Reads and merges config, registers models
```

3. **Use the ModelService or LMModule** to run inference:
```python
from src.ember.core.registry.model.core.modules.lm_modules import LMModule, LMModuleConfig

# Minimal usage with an LMModule:
config = LMModuleConfig(model_id="openai:gpt-4o", temperature=0.7)
lm = LMModule(config=config)
response_text = lm("Hello world!")
print("Got:", response_text)

# Or via ModelService:
from src.ember.core.registry.model.core.services.model_service import ModelService
service = ModelService(registry=GLOBAL_MODEL_REGISTRY, usage_service=None)
resp = service.invoke_model(model_id="openai:gpt-4o", prompt="Hello world!")
print("Got:", resp.data)
```

That's it! Optionally set `auto_discover=False` if you don't want to fetch remote model lists, and skip usage tracking if you don't need usage logs.

### API Reference

#### ModelRegistry
**Purpose:** Manages model instances and metadata.

**Key Methods:**
 - `register_model(model_info: ModelInfo)`: Registers a new model.
 - `get_model(model_id: str) -> BaseProviderModel`: Retrieves a model instance.
 - `list_models() -> List[str]`: Returns a list of registered model IDs.
 - `unregister_model(model_id: str) -> None`: Unregisters a model.

#### ModelService
**Purpose:** Facade for model invocation.

**Key Methods:**
 - `invoke_model(model_id: str, prompt: str, **kwargs) -> ChatResponse`: Synchronous model invocation.
 - `invoke_model_async(model_id: str, prompt: str, **kwargs) -> ChatResponse`: Asynchronous model invocation.
 - `get_model(model_id: str) -> BaseProviderModel`: Retrieves a specific model.

#### UsageService
**Purpose:** Tracks and aggregates model usage statistics.

**Key Methods:**
 - `add_usage_record(model_id: str, usage_stats: UsageStats)`: Adds a usage record.
 - `get_usage_summary(model_id: str) -> UsageSummary`: Retrieves aggregated usage information.
