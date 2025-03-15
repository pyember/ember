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
3. **Use `poetry shell` for interactive work** - Creates a subshell with the environment activated
4. **Use `poetry run` for single commands** - Runs a command in the environment without activation
5. **Be aware of Poetry's environment location** - By default, it's in `{cache-dir}/virtualenvs/`

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

If `poetry shell` fails:

```bash
# Alternative 1: Manually activate
source $(poetry env info --path)/bin/activate

# Alternative 2: Always use poetry run
poetry run python -c "import sys; print(sys.executable)"
```

### Path Issues

If Python can't find Ember modules:

```bash
# Run the setup_imports.py script
poetry run python setup_imports.py

# Or ensure you're running from the project root
cd /path/to/ember
poetry run python src/ember/examples/basic/minimal_example.py
```