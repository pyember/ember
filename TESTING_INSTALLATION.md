# Testing Ember Installation

This document outlines a systematic process for testing the Ember installation process in a clean environment. This is useful for verifying that the package can be installed and used by new users without any issues.

## Prerequisites

Before testing the installation, ensure you have the following:

- Python 3.9 or newer (3.10, 3.11, and 3.12 supported)
- Poetry 1.5.0 or newer (recommended)
- Access to a terminal/command prompt
- Internet connection to download packages

## Testing Process

### 1. Creating a Clean Environment

#### Using Python's venv 

```bash
# Create a new directory for testing
mkdir ember_test && cd ember_test

# Create a virtual environment with Python 3.9+
python3 -m venv test_env

# Activate the environment
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Using pyenv (recommended for managing multiple Python versions)

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

# Create and activate a virtual environment
python -m venv test_env
source test_env/bin/activate
```

#### Using Homebrew Python on macOS

```bash
# Install Python via Homebrew
brew install python@3.11

# Verify the installation
/opt/homebrew/bin/python3.11 --version

# Create a directory for testing
mkdir ember_test && cd ember_test

# Create a virtual environment with Homebrew Python
/opt/homebrew/bin/python3.11 -m venv test_env
source test_env/bin/activate
```

#### Using conda

```bash
# Create a new conda environment with Python 3.11
conda create -n ember_test python=3.11

# Activate the environment
conda activate ember_test
```

### 2. Installing Poetry

If not already installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or install within the virtual environment:

```bash
pip install poetry
```

### 3. Installing Ember

#### Option A: Install from PyPI

```bash
# Minimal installation (OpenAI only)
poetry add ember-ai -E minimal

# Full installation
# poetry add ember-ai -E all
```

#### Option B: Install from local repository

```bash
# Clone the repository if testing a local version
git clone https://github.com/pyember/ember.git
cd ember

# Install dependencies
poetry install --with dev

# You can also install in development mode
# poetry install
```

### 4. Testing the Installation

Run the minimal examples to verify the installation:

```bash
# For a PyPI installation
poetry run python -c "import ember; print(ember.__version__)"

# For a local repository installation, from ember/ember_test (step 1)
poetry run python ../src/ember/examples/basic/minimal_example.py
poetry run python ../src/ember/examples/basic/minimal_operator_example.py
```

### 5. Verification Checklist

- [ ] Python 3.11+ requirement is enforced
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
ERROR: Package 'ember-ai' requires a different Python: 3.9.6 not in '<3.13,>=3.11'
```

**Resolution**: Install Python 3.11 or newer and create a new virtual environment.

### Dependency Conflicts

If you encounter dependency resolution problems:

**Resolution**: 
```bash
# Update Poetry's lock file
poetry lock --no-update

# Clear Poetry's cache
poetry cache clear pypi --all
```

### Installation Speed

If installation is slow due to dependency resolution:

**Resolution**:
```bash
# Use the --no-dev flag if you don't need development dependencies
poetry install --no-dev
```

## Reporting Issues

If you encounter any issues during the installation testing process, please:

1. Document the exact steps to reproduce the issue
2. Include your environment details (OS, Python version, Poetry version)
3. Copy the complete error message
4. Report the issue on the [GitHub repository](https://github.com/pyember/ember/issues)