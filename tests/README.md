# Testing the Simplified Import Structure

We've simplified the import structure to make it more intuitive:

## New Import Patterns

```python
# Import the Operator base class
from ember.operator import Operator

# Import NON components
from ember.non import UniformEnsemble, JudgeSynthesis, Sequential

# Use them together
ensemble = UniformEnsemble(num_units=3, model_name="openai:gpt-4o")
judge = JudgeSynthesis(model_name="anthropic:claude-3-opus")
pipeline = Sequential(operators=[ensemble, judge])
```

This is a more intuitive structure than the previous deep imports.
