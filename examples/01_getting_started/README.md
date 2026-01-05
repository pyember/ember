# Getting Started with Ember

Welcome to Ember! This directory contains the essential first steps for new users.

## Overview

These examples will help you:
- Verify your Ember installation
- Make your first model calls
- Understand basic concepts
- Build confidence with the framework

## Examples in Order

1. **hello_world.py** - Verify Installation
   - Check that Ember is properly installed
   - Import core components
   - Create your first simple operator
   - Time: ~1 minute

2. **first_model_call.py** - Basic Model Interaction
   - Make your first LLM API call
   - Understand the models API
   - Handle responses
   - Time: ~2 minutes

3. **model_comparison.py** - Compare Different Models
   - Call multiple models
   - Compare responses
   - Understand model selection
   - Time: ~5 minutes

4. **basic_prompt_engineering.py** - Prompt Techniques
   - Use temperature settings
   - Add system prompts
   - Control output format
   - Time: ~5 minutes

## Prerequisites

Before running these examples, ensure you have:

1. **Python 3.11+** installed
2. **Ember** installed: `uv pip install -e .`
3. **Credentials** configured (for examples 2-4):
   ```bash
   ember setup
   # Or configure credentials directly:
   ember configure set providers.openai.api_key "your-key-here"
   # Optional for model_comparison.py:
   ember configure set providers.anthropic.api_key "your-key-here"
   ```

## Running the Examples

```bash
# Run each example
uv run python examples/01_getting_started/hello_world.py
uv run python examples/01_getting_started/first_model_call.py
# ... and so on
```

## Common Issues

### "Module not found" Error
Make sure you're running from the project root directory and Ember is installed.

### "API key not found" Error
Run `ember setup`, or set credentials via `ember configure set providers.<provider>.api_key ...`.

### "Connection timeout" Error
Check your internet connection and firewall settings.

## What's Next?

After completing these examples, move on to:
- **02_core_concepts/** - Learn about operators, types, and context
- **03_operators/** - Build custom operators
- **04_compound_ai/** - Create multi-model systems

## Key Takeaways

By the end of this section, you should understand:
- How to verify Ember is working
- How to make basic model calls
- How to compare different models
- Basic prompt engineering techniques

Happy learning.
