# Contributing to Ember

We welcome contributions that align with Ember's philosophy of simplicity and power.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/pyember/ember.git
cd ember

# Install dependencies with uv
uv sync --all-extras

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting and formatting
uv run ruff check . --fix
uv run ruff format .
```

## Code Standards

- **Style**: Google Python Style Guide with 100-character line limit
- **Types**: Strict typing; aim for mypy `--strict` cleanliness
- **Tests**: Maintain test coverage; add tests for new functionality
- **Docs**: Google docstring format with Args/Returns/Raises sections

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `uv run pytest`
4. Ensure type checking passes: `uv run mypy src/`
5. Ensure linting passes: `uv run ruff check .`
6. Submit a pull request with a clear description

### Commit Messages

Use conventional commits:
- `feat(models): add support for new provider`
- `fix(xcs): resolve parallelization issue`
- `docs(readme): update installation instructions`
- `test(operators): add edge case coverage`

## Design Principles

When contributing, keep these principles in mind:

1. **Simple by Default** - Basic usage requires no configuration
2. **Progressive Disclosure** - Complexity available when needed
3. **Composition Over Configuration** - Build complex from simple
4. **Explicit Over Magic** - No hidden behaviors or `__getattr__` tricks

## Areas for Contribution

- **New Providers**: Add support for additional LLM providers
- **Operators**: Create new composable operators for common patterns
- **Data Sources**: Add loaders for popular datasets
- **Documentation**: Improve examples and guides
- **Performance**: Optimize critical paths with profiling data

## Questions?

Open an issue for questions about contributing or to discuss larger changes before implementing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
