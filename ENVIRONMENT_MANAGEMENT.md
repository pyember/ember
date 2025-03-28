# Ember Environment Management Guide

This guide explains how to effectively manage Python environments when working with Ember using uv.

## Python Environment Management with uv

uv provides simplified Python environment management with these benefits:

- **Dependency Isolation**: Prevents conflicts between project dependencies
- **Reproducible Environments**: Ensures consistent behavior across development setups
- **Simplified Workflow**: Reduces the need for explicit environment activation

## Environment Management Approaches

### 1. Using uv's Simplified Environment Management (Recommended)

The simplest approach is to use uv's `run` command, which handles environments automatically:

```bash
# Install Ember
cd ember
uv pip install -e "."

# Run Python code without explicit environment activation
uv run python src/ember/examples/basic/minimal_example.py

# Run tools without explicit environment activation
uv run pytest
```

### 2. Traditional Virtual Environment Workflow

If you prefer a more traditional virtual environment workflow:

```bash
# Create a virtual environment in the project directory
uv venv

# Activate the environment (still required for interactive use)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Ember in the active environment
uv pip install -e "."

# Run code in the activated environment
python src/ember/examples/basic/minimal_example.py
```

### 3. Using Other Virtual Environment Tools

If you prefer using other environment managers:

```bash
# Create environment with venv
python -m venv ember_env
source ember_env/bin/activate  # On Windows: ember_env\Scripts\activate

# Install with uv in this environment
uv pip install -e "."
```

## Environment Management Best Practices

1. **Always use isolated environments** - Never install Ember in your global Python environment
2. **For simple usage, use `uv run`** - This handles environment management automatically
3. **For interactive shell work:**
   - Create a virtual environment with `uv venv`
   - Activate it with `source .venv/bin/activate`
4. **For running tools directly** - Use `uvx` which runs tools in isolated environments:
   ```bash
   uvx black src tests
   uvx mypy src
   ```

## Common Environment Commands

```bash
# Create a virtual environment in the current directory
uv venv

# Create a virtual environment with a specific Python version
uv venv --python=3.11

# Install packages
uv pip install -e "."
uv pip install -e ".[dev]"  # With development extras

# Run in an isolated environment
uv run python script.py
uv run pytest tests/
```

## Python Version Management

uv can also manage Python versions:

```bash
# Install Python versions
uv python install 3.10 3.11 3.12

# Use a specific Python version
uv venv --python 3.11
uv run --python 3.11 -- python script.py

# Pin a Python version for a project
uv python pin 3.11  # Creates .python-version
```

## Troubleshooting

### Python Version Issues

```bash
# Check Python version
python --version

# Specify a Python version for a virtual environment
uv venv --python 3.11

# Install a specific Python version with uv
uv python install 3.11
```

### Path Issues

If Python can't find Ember modules:

```bash
# Ensure you're running from the project root
cd /path/to/ember
uv run python src/ember/examples/basic/minimal_example.py
```

### Dependency Resolution Issues

If you encounter dependency conflicts:

```bash
# Use cached resolution if available
uv pip install -e "." --cache-only

# Force re-resolution
uv pip install -e "." --no-cache
```