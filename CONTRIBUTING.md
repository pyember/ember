# Contributing to Ember

Thank you for your interest in contributing to Ember! We're excited to welcome you to our community and appreciate your help in making Ember better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment](#development-environment)
  - [Running Tests](#running-tests)
  - [Code Style](#code-style)
- [Contribution Workflow](#contribution-workflow)
  - [Finding Issues](#finding-issues)
  - [Opening Issues](#opening-issues)
  - [Making Changes](#making-changes)
  - [Pull Requests](#pull-requests)
  - [Code Review](#code-review)
- [Development Guidelines](#development-guidelines)
  - [Documentation](#documentation)
  - [Testing](#testing)
  - [Code Quality](#code-quality)
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
   poetry install
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

### Running Tests

We use pytest for testing. To run the test suite:

```bash
# Run all tests
poetry run pytest

# Run specific tests
poetry run pytest tests/unit/core

# Run with coverage
poetry run pytest --cov=ember
```

### Code Style

We follow these guidelines:

- [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings
- [Black](https://black.readthedocs.io/) for automatic formatting
- [Type hints](https://www.python.org/dev/peps/pep-0484/) for function signatures

Before submitting code, please ensure it passes our linting checks:

```bash
# Run linting
poetry run pylint ember tests

# Format code
poetry run black ember tests
```

## Contribution Workflow

### Finding Issues

- Check our [issue tracker](https://github.com/foundrytechnologies/ember/issues) for open issues
- Look for issues tagged with [`good first issue`](https://github.com/foundrytechnologies/ember/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) if you're new to the project
- Feel free to ask questions in the issue comments if you need clarification

### Opening Issues

When opening a new issue, please:

- **Search existing issues** to avoid duplicates
- **Use a clear and descriptive title**
- For bug reports, include:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Environment details (OS, Python version, etc.)
- For feature requests, explain:
  - The problem you're trying to solve
  - Your proposed solution
  - Alternatives you've considered

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-commented code
   - Add/update tests for your changes
   - Update documentation as needed

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
4. **Ensure all tests pass** and code quality checks succeed
5. **Add reviewers** if you know who should review your code

### Code Review

During code review:

- Be responsive to feedback
- Make requested changes promptly
- Ask questions if something isn't clear
- Be patient and respectful

## Development Guidelines

### Documentation

Good documentation is essential:

- **Docstrings**: Add comprehensive docstrings to all public modules, classes, and functions
- **Examples**: Include usage examples for complex functionality
- **Comments**: Explain complex or non-obvious parts of your code
- **README updates**: Update relevant sections of README.md for significant changes

### Testing

We aim for high test coverage:

- **Unit tests**: For individual components and functions
- **Integration tests**: For interactions between components
- **Test edge cases**: Consider boundary conditions and error cases
- **Mocks and stubs**: Use them to isolate components during testing

### Code Quality

Maintain high code quality:

- **SOLID principles**: Follow solid design principles
- **DRY (Don't Repeat Yourself)**: Avoid code duplication
- **KISS (Keep It Simple, Stupid)**: Favor simplicity over complexity
- **Performance**: Consider the performance implications of your code

## Release Process

Our release process follows these steps:

1. Feature development in feature branches
2. Pull requests to the main branch
3. Periodic releases with semantic versioning

## Community

- **Discussions**: Join our [GitHub Discussions](https://github.com/foundrytechnologies/ember/discussions) for questions and ideas
- **Issues**: Use [GitHub Issues](https://github.com/foundrytechnologies/ember/issues) for bug reports and feature requests

---

Thank you for contributing to Ember! Your time and effort help make this project better for everyone.

## License

By contributing to Ember, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).