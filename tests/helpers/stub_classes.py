"""
Stub classes for tests.

This module provides stub implementations of key classes to avoid circular
dependencies during test collection and execution. These stubs implement
the minimal interface needed for tests to collect and run without requiring
the full implementations.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

# Import the EmberModel to properly type operators
from tests.helpers.ember_model import EmberModel

# Type variables for operator inputs/outputs - bound to EmberModel for proper type checking
T_in = TypeVar("T_in", bound=EmberModel)
T_out = TypeVar("T_out", bound=EmberModel)


@runtime_checkable
class OperatorProtocol(Protocol):
    """Protocol defining the operator interface."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Call method for operators."""
        ...


class Specification:
    """Simplified specification that doesn't perform validation."""

    def __init__(self, input_model=None, structured_output=None):
        self.input_model = input_model
        self.structured_output = structured_output
        self.prompt_template = None

    def validate_inputs(self, *, inputs):
        """No-op input validation."""
        if self.input_model and isinstance(inputs, dict):
            return self.input_model(**inputs)
        return inputs

    def validate_output(self, *, output):
        """No-op output validation."""
        return output

    def render_prompt(self, *, inputs):
        """Render a prompt from input."""
        if self.prompt_template:
            if hasattr(inputs, "dict"):
                return self.prompt_template.format(**inputs.dict())
            return self.prompt_template.format(**inputs)
        return str(inputs)


class EmberModule:
    """Simplified EmberModule stub."""

    def __init__(self, **kwargs):
        """Initialize EmberModule fields."""
        for key, value in kwargs.items():
            setattr(self, key, value)


class Operator(EmberModule, Generic[T_in, T_out]):
    """Base class for all operators."""

    specification: Optional[Specification] = None

    def __init__(self, **kwargs):
        """Initialize the operator."""
        super().__init__(**kwargs)

    def __call__(
        self, *, inputs: Union[T_in, Dict[str, Any]] = None, **kwargs
    ) -> T_out:
        """Call the operator's forward method."""
        if inputs is None:
            inputs = kwargs
        return self.forward(inputs=inputs)

    def forward(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Process inputs to produce outputs. Must be implemented by subclasses."""
        raise NotImplementedError()

    def _init_field(self, field_name: str, value: Any) -> None:
        """Initialize a field with a value."""
        setattr(self, field_name, value)


# Operator model stubs needed for tests


# Define model classes for NON Operators
class EnsembleInputs(EmberModel):
    """Input type for ensemble operators."""

    query: str


class EnsembleOutputs(EmberModel):
    """Output type for ensemble operators."""

    responses: List[str]


class MostCommonInputs(EmberModel):
    """Input type for MostCommon operator."""

    query: str
    responses: List[str]


class MostCommonOutputs(EmberModel):
    """Output type for MostCommon operator."""

    final_answer: str


class JudgeSynthesisInputs(EmberModel):
    """Input type for JudgeSynthesis operator."""

    query: str
    responses: List[str]


class JudgeSynthesisOutputs(EmberModel):
    """Output type for JudgeSynthesis operator."""

    synthesized_response: str
    reasoning: str


class VerifierInputs(EmberModel):
    """Input type for Verifier operator."""

    query: str
    candidate_answer: str


class VerifierOutputs(EmberModel):
    """Output type for Verifier operator."""

    verdict: str
    explanation: str
    revised_answer: str


class SequentialInputs(EmberModel):
    """Input type for Sequential operator."""

    query: str


class SequentialOutputs(EmberModel):
    """Output type for Sequential operator."""

    result: str


# NON Operator Stubs
class UniformEnsemble(Operator[EnsembleInputs, EnsembleOutputs]):
    """Stub implementation of UniformEnsemble."""

    def __init__(self, num_units=3, model_name=None, temperature=1.0, **kwargs):
        super().__init__()
        self.num_units = num_units
        self.model_name = model_name
        self.temperature = temperature
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        if isinstance(inputs, dict):
            return EnsembleOutputs(responses=["stub response"] * self.num_units)
        return EnsembleOutputs(responses=["stub response"] * self.num_units)


class MostCommon(Operator[MostCommonInputs, MostCommonOutputs]):
    """Stub implementation of MostCommon."""

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return MostCommonOutputs(final_answer="stub answer")


class JudgeSynthesis(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Stub implementation of JudgeSynthesis."""

    def __init__(self, model_name=None, temperature=0.7, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return JudgeSynthesisOutputs(
            synthesized_response="stub synthesis", reasoning="stub reasoning"
        )


class Verifier(Operator[VerifierInputs, VerifierOutputs]):
    """Stub implementation of Verifier."""

    def __init__(self, model_name=None, temperature=0.7, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return VerifierOutputs(
            verdict="valid",
            explanation="stub explanation",
            revised_answer="stub revision",
        )


class Sequential(Operator[SequentialInputs, SequentialOutputs]):
    """Stub implementation of Sequential."""

    def __init__(self, operators=None, **kwargs):
        super().__init__()
        self.operators = operators or []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return SequentialOutputs(result="stub sequential result")


# XCS Stubs
class XCSStub:
    """Stub implementation of XCS."""

    def jit(self, func=None, **kwargs):
        """JIT decorator."""
        return func if func is not None else lambda f: f

    def autograph(self, func=None, **kwargs):
        """Autograph decorator."""
        return func if func is not None else lambda f: f

    def execute(self, *args, **kwargs):
        """Execute a graph."""
        return {}

    def vmap(self, func=None, **kwargs):
        """Vectorized map decorator."""
        return func if func is not None else lambda f: f

    def pmap(self, func=None, **kwargs):
        """Parallel map decorator."""
        return func if func is not None else lambda f: f

    def mesh_sharded(self, func=None, **kwargs):
        """Mesh-sharded decorator."""
        return func if func is not None else lambda f: f


class DeviceMesh:
    """Stub implementation of DeviceMesh."""

    def __init__(self, devices=None, shape=None):
        self.devices = devices or []
        self.shape = shape or (len(self.devices),)


class PartitionSpec:
    """Stub implementation of PartitionSpec."""

    def __init__(self, *mesh_axes):
        self.mesh_axes = mesh_axes


# Model Service Stubs
class ModelService:
    """Stub implementation of ModelService."""

    def __init__(self):
        pass

    async def chat(self, messages, model_name=None, **kwargs):
        """Chat method for model service."""
        return {"content": "stub response"}

    async def complete(self, prompt, model_name=None, **kwargs):
        """Complete method for model service."""
        return {"text": "stub completion"}


class UsageService:
    """Stub implementation of UsageService."""

    def __init__(self):
        pass

    def record_usage(self, usage_data):
        """Record usage method."""
        pass

    def get_usage_summary(self):
        """Get usage summary method."""
        return {"total_tokens": 0, "total_cost": 0.0}


# LM Module stubs
class LMModuleConfig:
    """Stub configuration for LMModule."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens


class LMModule:
    """Simplified LMModule stub."""

    def __init__(self, config: LMModuleConfig):
        self.config = config

    def __call__(self, prompt: str) -> str:
        """Generate a response for the prompt."""
        return f"Response to: {prompt}"


# Stubs for operator core classes
class EnsembleOperator(Operator[EnsembleInputs, EnsembleOutputs]):
    """Stub EnsembleOperator for tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return EnsembleOutputs(responses=["stub response"] * 3)


class MostCommonAnswerSelectorOperator(Operator[MostCommonInputs, MostCommonOutputs]):
    """Stub MostCommonAnswerSelectorOperator for tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return MostCommonOutputs(final_answer="stub answer")


class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Stub JudgeSynthesisOperator for tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return JudgeSynthesisOutputs(
            synthesized_response="stub synthesis", reasoning="stub reasoning"
        )


class VerifierOperator(Operator[VerifierInputs, VerifierOutputs]):
    """Stub VerifierOperator for tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, *, inputs):
        """Forward method for the operator."""
        return VerifierOutputs(
            verdict="valid",
            explanation="stub explanation",
            revised_answer="stub revision",
        )


# Define stub input types for operators
class EnsembleOperatorInputs(EmberModel):
    """Stub EnsembleOperatorInputs for tests."""

    query: str


class MostCommonAnswerSelectorOperatorInputs(EmberModel):
    """Stub MostCommonAnswerSelectorOperatorInputs for tests."""

    query: str
    responses: List[str]


class VerifierOperatorInputs(EmberModel):
    """Stub VerifierOperatorInputs for tests."""

    query: str
    candidate_answer: str


# Export everything for easy access
__all__ = [
    "Operator",
    "EmberModule",
    "T_in",
    "T_out",
    "Specification",
    "UniformEnsemble",
    "MostCommon",
    "JudgeSynthesis",
    "Verifier",
    "Sequential",
    "EnsembleInputs",
    "EnsembleOutputs",
    "MostCommonInputs",
    "MostCommonOutputs",
    "JudgeSynthesisInputs",
    "JudgeSynthesisOutputs",
    "VerifierInputs",
    "VerifierOutputs",
    "SequentialInputs",
    "SequentialOutputs",
    "ModelService",
    "UsageService",
    "EnsembleOperator",
    "MostCommonAnswerSelectorOperator",
    "JudgeSynthesisOperator",
    "VerifierOperator",
    "EnsembleOperatorInputs",
    "MostCommonAnswerSelectorOperatorInputs",
    "VerifierOperatorInputs",
    "XCSStub",
    "DeviceMesh",
    "PartitionSpec",
    "LMModule",
    "LMModuleConfig",
    "OperatorProtocol",
]
