# Example tests

This directory contains smoke tests for the example scripts in `examples/`.

Tests verify that each example script runs successfully without errors.
The examples themselves contain real, runnable code that produces output
when executed.

## Running tests

```bash
uv run pytest tests/examples -q
# or:
python -m pytest tests/examples -q
```

## Test structure

Each `test_XX_*.py` file corresponds to an example directory:

- `test_01_getting_started.py` - tests for `01_getting_started/`
- `test_02_core_concepts.py` - tests for `02_core_concepts/`
- etc.

Tests inherit from `ExampleTest` in `test_base.py`, which provides:

- `run_example(path, timeout=...)` - executes an example and captures output
- `run_example_test(path, timeout=...)` - runs and asserts success (exit code 0)

## Adding new examples

1. Create the example script in the appropriate `examples/XX_*/` directory.
2. Add a test method in the corresponding `test_XX_*.py` file:

```python
def test_my_new_example(self) -> None:
    self.run_example_test("XX_category/my_new_example.py")
```
