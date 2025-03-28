"""
Minimal test doubles for operator components.

This module provides simplified test doubles that implement just enough functionality
to test client code without duplicating the implementation. Following the
principle of "avoid overmocking" from CLAUDE.md guidelines.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, List, TypeVar

# Type variables for operator inputs/outputs
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class MinimalOperator(Generic[T_in, T_out]):
    """Minimal test double for the Operator base class."""

    def __init__(self):
        """Initialize with minimal required state."""
        pass

    def __call__(self, *, inputs: T_in) -> T_out:
        """Delegate to forward method."""
        return self.forward(inputs=inputs)

    def forward(self, *, inputs: T_in) -> T_out:
        """Default implementation that must be overridden."""
        raise NotImplementedError("Subclasses must implement forward method")


@dataclass
class MinimalTestModel:
    """A simple model for testing."""

    value: Any = None


class SimpleLLMOperator(MinimalOperator[MinimalTestModel, MinimalTestModel]):
    """A simple LLM-based operator for testing."""

    def __init__(self, response_text: str = "This is a test response"):
        """Initialize with a fixed response."""
        super().__init__()
        self.response_text = response_text

    def forward(self, *, inputs: MinimalTestModel) -> MinimalTestModel:
        """Return a fixed response regardless of input."""
        return MinimalTestModel(value=self.response_text)


class SimpleDeterministicOperator(MinimalOperator[MinimalTestModel, MinimalTestModel]):
    """A simple deterministic operator for testing."""

    def __init__(self, transform_fn: Callable[[Any], Any] = lambda x: x):
        """Initialize with a transform function."""
        super().__init__()
        self.transform_fn = transform_fn

    def forward(self, *, inputs: MinimalTestModel) -> MinimalTestModel:
        """Apply the transform function to the input value."""
        return MinimalTestModel(value=self.transform_fn(inputs.value))


class SimpleEnsembleOperator(MinimalOperator[MinimalTestModel, MinimalTestModel]):
    """A simple ensemble operator for testing."""

    def __init__(self, operators: List[MinimalOperator] = None):
        """Initialize with a list of operators."""
        super().__init__()
        self.operators = operators or []

    def forward(self, *, inputs: MinimalTestModel) -> MinimalTestModel:
        """Execute all operators and return a list of results."""
        results = [op(inputs=inputs).value for op in self.operators]
        return MinimalTestModel(value=results)


class SimpleSelectorOperator(MinimalOperator[MinimalTestModel, MinimalTestModel]):
    """A simple selector operator for testing."""

    def __init__(self, operators: List[MinimalOperator] = None, select_index: int = 0):
        """Initialize with operators and selection index."""
        super().__init__()
        self.operators = operators or []
        self.select_index = select_index

    def forward(self, *, inputs: MinimalTestModel) -> MinimalTestModel:
        """Execute all operators and select one result."""
        if not self.operators:
            return MinimalTestModel(value=None)

        results = [op(inputs=inputs).value for op in self.operators]
        selected = results[min(self.select_index, len(results) - 1)]
        return MinimalTestModel(value=selected)


# Export minimal test doubles
__all__ = [
    "MinimalOperator",
    "MinimalTestModel",
    "SimpleLLMOperator",
    "SimpleDeterministicOperator",
    "SimpleEnsembleOperator",
    "SimpleSelectorOperator",
]
