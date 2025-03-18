# Ember Installation Guide

This guide provides detailed instructions for installing Ember in different environments.

## System Requirements

- **Python**: 3.9 or newer (3.10, 3.11, and 3.12 supported)
- **Operating System**: macOS, Linux, or Windows
- **Package Manager**: Poetry 1.5.0 or newer

## Installation Methods

### Method 1: Using Poetry (Recommended)

Poetry is the recommended package manager for Ember. It automatically creates isolated environments and manages dependencies.

1. **Install Poetry** if you don't have it already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   After installation, make sure Poetry is in your PATH:
   ```bash
   # Add Poetry to your PATH (add this to your .bashrc or .zshrc for persistence)
   export PATH="$HOME/.local/bin:$PATH"
   
   # Verify the installation
   poetry --version
   ```

2. **Clone and install Ember**:
   ```bash
   # Clone the repository
   git clone https://github.com/pyember/ember.git
   cd ember
   
   # Install with Poetry (creates a virtual environment automatically)
   poetry install
   
   # Activate the Poetry-managed virtual environment
   # For Poetry 2.0+
   poetry env use python3
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # For Poetry 1.x
   poetry shell
   
   # Alternatively, run commands directly within the environment
   poetry run python src/ember/examples/basic/minimal_example.py
   ```
   
   By default, this installs Ember with OpenAI, Anthropic, and Google/Deepmind provider support.

### Method 2: Development Installation

If you want to develop or contribute to Ember:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pyember/ember.git
   cd ember
   ```

2. **Install with development dependencies**:
   ```bash
   # Install including development dependencies
   poetry install --with dev
   
   # Activate the Poetry environment
   poetry shell
   ```

3. **Setup correct import paths** (optional but recommended):
   ```bash
   # Run the setup script to ensure imports work correctly
   python setup_imports.py
   ```

## OS-Specific Installation Notes

### macOS

On macOS, you might encounter issues with the default Python installation:

```bash
# If you get a "cannot create venvs without using symlinks" error:
# Install Python using Homebrew (recommended)
brew install python@3.11

# Use the Homebrew Python with Poetry
poetry env use $(brew --prefix python@3.11)/bin/python3.11
```

### Windows

On Windows, ensure you have the latest Python installed from python.org:

```powershell
# Add Poetry to your PATH
$env:PATH += ";$env:USERPROFILE\.poetry\bin"

# Use PowerShell for activation
poetry shell
```

## Troubleshooting

### Python Version Issues

If you encounter Python version errors:

```bash
# Check your Python version
python --version

# If using the wrong version, specify the correct Python for Poetry
poetry env use python3.11
# Or specify the exact path
poetry env use /path/to/python3.11
```

### Poetry Installation Issues

If you have problems with Poetry:

```bash
# Ensure Poetry is in your PATH
echo $PATH

# Upgrade Poetry
poetry self update

# Clear Poetry's cache
poetry cache clear pypi --all
```

### Virtual Environment Issues

If you have problems with virtual environments:

```bash
# See what environment Poetry is using
poetry env info

# Create a new environment with a specific Python version
poetry env use python3.11
```

See [ENVIRONMENT_MANAGEMENT.md](ENVIRONMENT_MANAGEMENT.md) for more details on managing environments.

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Update Poetry lock file
poetry lock --no-update

# Install with verbose output
poetry install -v
```

## Testing Your Installation

After installation, verify everything is working:

```bash
# From the project root directory, using the Poetry environment
poetry run python src/ember/examples/basic/minimal_example.py

# Or if you're in a Poetry shell
python src/ember/examples/basic/minimal_example.py
```

## Getting Help

If you encounter issues with installation:
- Check our [GitHub Issues](https://github.com/pyember/ember/issues)
- Review the [ENVIRONMENT_MANAGEMENT.md](ENVIRONMENT_MANAGEMENT.md) guide
- See the [TESTING_INSTALLATION.md](TESTING_INSTALLATION.md) for verification steps