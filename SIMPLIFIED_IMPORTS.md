# Simplified Ember Import Structure

This PR simplifies the import structure for operators and NON functionality by creating:

1. A top-level `ember.operator` module providing direct access to the base Operator class
2. A top-level `ember.non` module for accessing NON components

This enables cleaner imports like:

```python
from ember.operator import Operator
from ember.non import UniformEnsemble, Sequential
```

The implementation follows SOLID principles and maintains backward compatibility.
