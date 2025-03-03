"""
Test script to examine operator return handling
"""

from typing import List, Dict, Any, Type
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.prompt_specification.specification import Specification
from ember.core.types import EmberModel


class OperatorTestInputs(EmberModel):
    query: str


class OperatorTestOutputs(EmberModel):
    results: List[str]


class SimpleSpecification(Specification):
    prompt_template: str = "Test prompt: {query}"
    input_model: Type[EmberModel] = OperatorTestInputs
    output_model: Type[EmberModel] = OperatorTestOutputs


class SimpleOperator(Operator[OperatorTestInputs, OperatorTestOutputs]):
    specification = SimpleSpecification()

    def forward(self, *, inputs: OperatorTestInputs) -> Dict[str, Any]:
        # Return a dictionary instead of a model instance
        return {"results": ["test1", "test2"]}


# Create and call the operator
op = SimpleOperator()
inputs = OperatorTestInputs(query="test query")
result = op(inputs=inputs)

# Examine the result
print(f"Result type: {type(result)}")
print(f"Is dict: {isinstance(result, dict)}")
print(f"Is EmberModel: {hasattr(result, 'model_dump')}")
print(f"Has results attribute: {hasattr(result, 'results')}")

if hasattr(result, "results"):
    print(f"Results: {result.results}")

# Print all attributes
print("\nAll attributes:")
for attr in dir(result):
    if not attr.startswith("__") and not callable(getattr(result, attr)):
        try:
            value = getattr(result, attr)
            print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: <error: {e}>")
