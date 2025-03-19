# Ember Environment Management Guide

This guide explains how to effectively manage Python environments when working with Ember, especially focusing on Poetry's virtual environment capabilities.

## Understanding Poetry's Environment Management

Poetry automatically creates and manages isolated virtual environments for your projects, providing several benefits:

- **Dependency Isolation**: Prevent conflicts between project dependencies
- **Reproducible Environments**: Ensure consistent behavior across development setups
- **Clean Environment Management**: Automated creation and activation of virtual environments

## Environment Management Approaches

### 1. Using Poetry's Built-in Environment Management (Recommended)

Poetry creates and manages virtual environments automatically:

```bash
# Install Ember with Poetry (creates the environment automatically)
cd ember
poetry install

# Enter the virtual environment
# For Poetry 2.0+
poetry env use python3
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# For Poetry 1.x
poetry shell

# Run commands within the environment without activation
poetry run python src/ember/examples/basic/minimal_example.py
```

### 2. Using External Virtual Environment Tools

If you prefer using other environment managers:

```bash
# Create environment with venv
python -m venv ember_env
source ember_env/bin/activate  # On Windows: ember_env\Scripts\activate

# Install with pip in this environment
pip install -e .

# Or use Poetry with an existing environment
poetry config virtualenvs.create false  # Tell Poetry to use the current environment
poetry install
```

## Environment Management Best Practices

1. **Always use virtual environments** - Never install Ember in your global Python environment
2. **Let Poetry handle environments when possible** - It manages dependency resolution better
3. **For interactive work**:
   - **Poetry 2.0+**: Use `poetry env use python3` followed by `source .venv/bin/activate`
   - **Poetry 1.x**: Use `poetry shell` which creates a subshell with the environment activated
4. **Use `poetry run` for single commands** - Runs a command in the environment without activation
5. **Be aware of Poetry's environment location** - By default, it's in `{cache-dir}/virtualenvs/` or in a local `.venv` directory

## Common Environment Commands

```bash
# See where Poetry stores your virtual environments
poetry config virtualenvs.path

# Create a new Poetry environment
poetry env use python3.11

# List Poetry environments
poetry env list

# Show information about the current environment
poetry env info

# Remove a Poetry environment
poetry env remove <environment-name>
```

## Troubleshooting

### Poetry Can't Find Python Version

```bash
# Check available Python versions
poetry env use --help

# Specify a specific Python executable
poetry env use /path/to/python

# Install additional Python version
# macOS: brew install python@3.11
# Linux: apt install python3.11 (or use pyenv)
```

### Environment Activation Issues

#### Poetry 2.0+ Environments

In Poetry 2.0+, the `shell` command is not installed by default. Instead, use:

```bash
# Set up the environment with your Python interpreter
poetry env use python3

# Activate the environment directly (recommended)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or get the activation command from Poetry
source $(poetry env info --path)/bin/activate

# Check that you're in the correct environment
which python  # Should point to the Poetry environment
```

#### If `poetry shell` Fails in Poetry 1.x:

```bash
# Alternative 1: Manually activate
source $(poetry env info --path)/bin/activate

# Alternative 2: Always use poetry run
poetry run python -c "import sys; print(sys.executable)"
```

### Path Issues

If Python can't find Ember modules:

```bash
# Ensure you're running from the project root
cd /path/to/ember
poetry run python src/ember/examples/basic/minimal_example.py
```