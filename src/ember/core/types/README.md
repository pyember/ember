# EmberModel Type System

## Overview

EmberModel is a unified type system for Ember that standardizes input/output models across all operators. It combines the validation capabilities of Pydantic with flexible serialization options to maintain backward compatibility.

## Features

- **Strong typing**: Full static type checking with mypy
- **Validation**: Built-in validation through Pydantic
- **Flexibility**: Can behave like a BaseModel or TypedDict based on configuration
- **Serialization**: Easy conversion to dict, JSON, and other formats
- **Backward compatibility**: Works with existing code expecting different formats

## Usage Guidelines

### Defining Models

```python
from ember.core.types import EmberModel

class MyOperatorInputs(EmberModel):
    query: str
    max_tokens: int = 100
```

### Using Models in Operators

```python
class MyOperator(Operator[MyOperatorInputs, MyOperatorOutputs]):
    def forward(self, *, inputs: MyOperatorInputs) -> MyOperatorOutputs:
        # Access fields directly
        query = inputs.query
        
        # Convert to dict if needed for backward compatibility
        inputs_dict = inputs.as_dict()
        
        # Return a dict literal - EmberModel will handle the conversion
        return {"result": "Example result"}
```

### Controlling Format Behavior

You can control how a model behaves when used as a function:

```python
# Get a model instance (default behavior)
result_model = model()

# Get as dictionary
model.set_output_format("dict")
result_dict = model()

# Get as JSON
model.set_output_format("json")
result_json = model()
```

### Dictionary Access Compatibility

EmberModel supports dictionary-like access for backward compatibility:

```python
# These are equivalent:
model.field_name
model["field_name"]
```

## Compatibility with Existing Code

EmberModel was designed for minimal impact on existing code:

1. Existing code that returns plain dictionaries in operators will still work
2. Code that expects dict inputs/outputs will still work with EmberModel instances
3. Validation and type safety is improved without breaking changes

## Best Practices

1. Use EmberModel for all new operator input/output types
2. Keep returning dicts from operator.forward() methods for backward compatibility
3. Use typed access (e.g., `model.field`) when possible for better IDE support
4. Let EmberModel handle the conversion between typed and untyped formats