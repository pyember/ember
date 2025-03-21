# Testing Ember Installation

This document outlines a systematic process for testing the Ember installation process in a clean environment. This is useful for verifying that the package can be installed and used by new users without any issues.

## Prerequisites

Before testing the installation, ensure you have the following:

- Python 3.9 or newer (3.10, 3.11, and 3.12 supported)
- uv installed (recommended) or any Python package manager
- Access to a terminal/command prompt
- Internet connection to download packages

## Testing Process

### 1. Creating a Clean Environment

#### Using uv (recommended)

```bash
# Create a new directory for testing
mkdir ember_test && cd ember_test

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create a virtual environment with uv
uv venv

# Activate the environment (only needed for interactive use)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Using Python's venv 

```bash
# Create a new directory for testing
mkdir ember_test && cd ember_test

# Create a virtual environment with Python 3.9+
python3 -m venv test_env

# Activate the environment
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install uv in the environment
pip install uv
```

#### Using pyenv for Python version management

```bash
# Install pyenv if not already installed
# macOS: brew install pyenv
# Linux: curl https://pyenv.run | bash

# Install Python using pyenv (3.9, 3.10, 3.11, or 3.12)
pyenv install 3.11.x

# Create a directory for testing
mkdir ember_test && cd ember_test

# Set local Python version
pyenv local 3.11.x

# Use uv to create a virtual environment
uv venv
source .venv/bin/activate
```

#### Using Homebrew Python on macOS

```bash
# Install Python via Homebrew
brew install python@3.11

# Verify the installation
/opt/homebrew/bin/python3.11 --version

# Create a directory for testing
mkdir ember_test && cd ember_test

# Use the Homebrew Python with uv
/opt/homebrew/bin/python3.11 -m pip install uv
/opt/homebrew/bin/python3.11 -m uv venv
source .venv/bin/activate
```

#### Using conda

```bash
# Create a new conda environment with Python 3.11
conda create -n ember_test python=3.11

# Activate the environment
conda activate ember_test

# Install uv in the conda environment
pip install uv
```

### 2. Installing Ember

#### Option A: Install from PyPI

```bash
# Minimal installation (OpenAI only)
uv pip install "ember-ai[minimal]"

# Full installation
# uv pip install "ember-ai[all]"
```

#### Option B: Install from local repository

```bash
# Clone the repository if testing a local version
git clone https://github.com/pyember/ember.git
cd ember

# Install dependencies with development extras
uv pip install -e ".[dev]"
```

### 3. Testing the Installation

Run the minimal examples to verify the installation:

```bash
# Check if the package imports correctly
uv run python -c "import ember; print(ember.__version__)"

# For a local repository installation
uv run python src/ember/examples/basic/minimal_example.py
uv run python src/ember/examples/basic/minimal_operator_example.py

# Or if you're in an activated environment
python -c "import ember; print(ember.__version__)"
python src/ember/examples/basic/minimal_example.py
```

### 4. Verification Checklist

- [ ] Python 3.9+ requirement is enforced
- [ ] All dependencies are correctly resolved
- [ ] Core LLM providers (OpenAI, Anthropic, Google/Deepmind) are installed
- [ ] No errors during installation process
- [ ] Examples run without errors
- [ ] Import statements work correctly
- [ ] Basic functionality is operational

## Common Issues and Resolutions

### Python Version

If you encounter errors related to Python version compatibility:

```
ERROR: Package 'ember-ai' requires a different Python: 3.9.6 not in '<3.13,>=3.9'
```

**Resolution**: Install Python 3.9 or newer and create a new virtual environment.

### Dependency Conflicts

If you encounter dependency resolution problems:

**Resolution**: 
```bash
# Try reinstalling with no cache to force re-resolution
uv pip install -e "." --no-cache

# Or specify exact versions if needed
uv pip install -e "." --no-deps
uv pip install "dependency==specific.version"
```

### Installation Speed

If you're experiencing slow installation (unlikely with uv):

**Resolution**:
```bash
# Use the minimal installation if you only need basic functionality
uv pip install "ember-ai[minimal]"
```

## Reporting Issues

If you encounter any issues during the installation testing process, please:

1. Document the exact steps to reproduce the issue
2. Include your environment details (OS, Python version, uv version)
3. Copy the complete error message
4. Report the issue on the [GitHub repository](https://github.com/pyember/ember/issues)