# Contributing to Ember

Thank you for your interest in contributing to Ember! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment](#development-environment)
  - [Project Structure](#project-structure)
  - [Running Tests](#running-tests)
  - [Code Style and Quality](#code-style-and-quality)
- [Contribution Workflow](#contribution-workflow)
  - [Finding Issues](#finding-issues)
  - [Creating Issues](#creating-issues)
  - [Making Changes](#making-changes)
  - [Pull Requests](#pull-requests)
  - [Code Review](#code-review)
- [Development Guidelines](#development-guidelines)
  - [Documentation](#documentation)
  - [Testing](#testing)
  - [Performance Considerations](#performance-considerations)
  - [Typed Code](#typed-code)
- [Release Process](#release-process)
- [Community](#community)

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ember.git
   cd ember
   ```

2. **Set up Poetry (recommended)**:
   We use Poetry for dependency management. [Install Poetry](https://python-poetry.org/docs/#installation) if you haven't already.

3. **Install dependencies**:
   ```bash
   # Install with all development dependencies
   poetry install --with dev
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

5. **Set up pre-commit hooks** (recommended):
   ```bash
   pre-commit install
   ```

#### Alternative Installation with pip

If you prefer not to use Poetry, you can use pip:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

#### Note on Imports

The project is set up with proper Python packaging, so you should import from `ember` directly:

```python
# Correct way to import
from ember.core import non
from ember.xcs.tracer import jit

# No need to manipulate sys.path or use symlinks
```

### Project Structure

The Ember codebase is organized into the following structure:

```
ember/
├── docs/               # Documentation
│   ├── cli/            # CLI documentation
│   ├── design/         # Design documents
│   └── quickstart/     # Quick start guides
├── src/                # Source code
│   ├── cli/            # TypeScript CLI implementation
│   └── ember/          # Main Python package
│       ├── cli.py      # Python CLI entrypoint
│       ├── core/       # Core framework
│       ├── xcs/        # Execution engine
│       └── examples/   # Example applications
├── tests/              # Test suite
│   ├── cli/            # CLI tests
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── fuzzing/        # Fuzzing tests
├── examples/           # Standalone examples 
├── scripts/            # Development and CI scripts
├── pyproject.toml      # Python project configuration
├── package.json        # Node.js project configuration
├── tsconfig.json       # TypeScript configuration
└── README.md           # Project overview
```

The `.gitignore` file is configured to exclude common development files, caches, and sensitive configuration files.

### Running Tests

We use pytest for testing. To run the test suite:

```bash
# Run all tests
poetry run pytest

# Run specific tests
poetry run pytest tests/unit/core

# Run tests with code coverage
poetry run pytest --cov=src/ember

# Run a specific test file
poetry run pytest tests/unit/core/test_app_context.py
```

### Code Style and Quality

We enforce high code quality standards:

1. **Code Formatting**:
   - We use Black for code formatting
   - Line length is set to 88 characters
   - Run `poetry run black src tests` before committing

2. **Import Sorting**:
   - We use isort for import sorting
   - Run `poetry run isort src tests` before committing

3. **Linting**:
   - We use ruff and pylint for linting
   - Run `poetry run ruff check src tests` before committing
   - Run `poetry run pylint src/ember` for more detailed linting

4. **Type Checking**:
   - We use mypy for static type checking
   - Run `poetry run mypy src` before committing

All these checks are also performed automatically when you submit a pull request.

## Contribution Workflow

### Finding Issues

- Check our [issue tracker](https://github.com/pyember/ember/issues) for open issues
- Look for issues tagged with [`good first issue`](https://github.com/pyember/ember/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) if you're new to the project
- Feel free to ask questions in the issue comments if you need clarification

### Creating Issues

When opening a new issue, please:

- **Search existing issues** first to avoid duplicates
- **Use a clear and descriptive title**
- **Follow the issue template** if one is provided
- For bug reports, include:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Environment details (OS, Python version, etc.)
  - Code samples or error traces when relevant
- For feature requests, explain:
  - The problem you're trying to solve
  - Your proposed solution
  - Alternatives you've considered

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-commented code
   - Add/update tests to cover your changes
   - Update documentation as needed
   - Ensure your code passes all tests and style checks

3. **Commit your changes**:
   - Use clear, meaningful commit messages
   - Reference issue numbers where applicable
   ```bash
   git commit -m "Add feature X, fixes #123"
   ```

4. **Keep your branch updated**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

### Pull Requests

When submitting a pull request:

1. **Fill out the PR template** completely
2. **Link to related issues**
3. **Describe your changes** in detail
4. **Ensure all tests and checks pass**
5. **Include screenshots or examples** for UI or behavior changes
6. **Request reviews** from maintainers or contributors familiar with the area of code

### Code Review

During code review:

- Be responsive to feedback
- Make requested changes promptly
- Ask questions if something isn't clear
- Be patient and respectful
- Remember that the goal is to improve code quality

## Development Guidelines

### Documentation

Good documentation is essential:

1. **Docstrings**:
   - All public modules, classes, and functions must have docstrings
   - We follow Google-style docstrings
   - Include type hints in docstrings for complex parameters
   - Example:
   ```python
   def process_data(data: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Result:
       """Process input data with optional configuration.

       Args:
           data: List of data dictionaries to process
           options: Optional configuration parameters

       Returns:
           A Result object containing processed output

       Raises:
           ValueError: If data is empty or malformed
       """
   ```

2. **README and Documentation Files**:
   - Update relevant documentation for significant changes
   - Keep examples up-to-date
   - Add new documentation for new features

3. **Code Comments**:
   - Use comments for complex or non-obvious logic
   - Avoid redundant comments that just restate the code
   - Use TODO comments for future improvements (with issue references)

### Testing

We strive for high test coverage:

1. **Test Coverage**:
   - All new code should have corresponding tests
   - We aim for at least 90% code coverage
   - Critical paths should have 100% coverage

2. **Test Types**:
   - **Unit tests**: For testing individual functions and classes in isolation
   - **Integration tests**: For testing interactions between components
   - **Property-based tests**: Using Hypothesis for testing invariants
   - **Fuzzing tests**: For finding edge cases and security issues

3. **Test Naming and Organization**:
   - Test files should be named `test_*.py`
   - Test classes should be named `Test*`
   - Test functions should be named `test_*`
   - Group related tests in the same file or directory

4. **Test Quality**:
   - Tests should be deterministic and reliable
   - Mock external dependencies appropriately
   - Test edge cases and error conditions
   - Include both positive and negative test cases

### Performance Considerations

Performance is important in Ember:

1. **Measurement**:
   - Use profiling tools to identify bottlenecks
   - Include benchmarks for performance-critical code
   - Compare before/after performance for optimizations

2. **Optimizations**:
   - Optimize for readability and maintainability first
   - Focus optimizations on critical paths
   - Document performance trade-offs in comments
   - Use appropriate data structures and algorithms

3. **Concurrency**:
   - Ensure thread safety for shared resources
   - Use appropriate locking mechanisms
   - Consider asynchronous approaches where applicable

### Typed Code

We use Python type hints extensively:

1. **Type Annotations**:
   - Annotate all function parameters and return values
   - Use appropriate generic types when needed
   - Use Optional, Union, and other typing constructs as needed

2. **Type Checking**:
   - Run `mypy` to check for type errors
   - Address all type warnings
   - Use TypeVar and Generic for polymorphic code

3. **Custom Types**:
   - Define new type aliases for complex types
   - Use Protocol for structural typing
   - Document type parameters and constraints

## Release Process

Our release process follows these steps:

1. Feature development in feature branches
2. Pull requests to the main branch after code review
3. Continuous integration tests on all PRs
4. Periodic releases with semantic versioning:
   - MAJOR version for incompatible API changes
   - MINOR version for backwards-compatible functionality
   - PATCH version for backwards-compatible bug fixes
5. Release notes summarizing changes and upgrades

## Community

- **Discussions**: Join our [GitHub Discussions](https://github.com/pyember/ember/discussions) for questions and ideas
- **Issues**: Use [GitHub Issues](https://github.com/pyember/ember/issues) for bug reports and feature requests
- **Discord**: Join our [Discord server](https://discord.gg/ember-ai) for real-time discussion

---

Thank you for contributing to Ember! Your time and effort help make this project better for everyone.

## License

By contributing to Ember, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).