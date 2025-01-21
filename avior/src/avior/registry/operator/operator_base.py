from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from pydantic import BaseModel
from src.avior.registry.prompt_signature.signatures import Signature
from src.avior.modules.lm_modules import LMModule
from src.avior.core.scheduler import Scheduler, ExecutionPlan


class OperatorType(Enum):
    """Enumeration representing operator topologies or roles."""
    RECURRENT = auto()
    FAN_OUT = auto()
    FAN_IN = auto()



class OperatorMetadata(BaseModel):
    """Metadata for an operator (useful for introspection and registry)."""
    code: str
    description: str
    operator_type: OperatorType
    signature: Optional[Signature] = None


def run_in_parallel(
    function, args_list: List[Dict[str, Any]], max_workers: int = None
) -> List[Any]:
    """Utility to run tasks in parallel and return results."""
    from concurrent.futures import ThreadPoolExecutor

    results: List[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(function, **args) for args in args_list]
        for future in futures:
            results.append(future.result())
    return results

T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class Operator(ABC, Generic[T_in, T_out]):
    """
    Base Operator that automatically discovers sub-operators.
    Focus: concurrency plan and forward execution.
    Validation delegated to the associated Signature.
    """

    # Each subclass typically overrides metadata with its own code, signature, etc.
    metadata: OperatorMetadata = OperatorMetadata(
        code="BASE",
        description="Base operator with sub-operator auto-registration.",
        operator_type=OperatorType.RECURRENT,
        signature=Signature(required_inputs=[]),
    )

    def __init__(
        self,
        name: str = "Operator",
        lm_modules: Optional[List[LMModule]] = None,
        sub_operators: Optional[List["Operator"]] = None,
        signature: Optional[Signature] = None,
    ):
        """
        :param name: A human-readable name for debugging or introspection.
        :param lm_modules: A list of LMModule instances for language model calls.
        :param sub_operators: Optional list of child operators to be auto-registered.
        :param signature: Optional signature for input/output validation.
        """
        # We bypass normal __setattr__ here to ensure we can create _sub_operators first
        object.__setattr__(self, "_sub_operators", {})
        self.name = name
        self.lm_modules = lm_modules if lm_modules else []
        # Conditionally override the signature only if explicitly provided:
        if signature is not None:
            self.metadata.signature = signature

        # Optionally register sub_operators
        for i, op in enumerate(sub_operators or []):
            self.__setattr__(f"sub_op_{i}", op)

    def __setattr__(self, attr_name: str, value: Any) -> None:
        """
        Auto-registration: if 'value' is an Operator and the attr_name isn't '__call__',
        store it in _sub_operators. Otherwise, set normally.
        """
        if attr_name != "__call__" and isinstance(value, Operator):
            self._sub_operators[attr_name] = value
        super().__setattr__(attr_name, value)

    @property
    def sub_operators(self) -> Dict[str, "Operator"]:
        """
        Returns a dictionary of discovered child operators keyed by their attribute name.
        """
        return getattr(self, "_sub_operators", {})

    def build_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Helper to create a prompt from inputs, if a signature + prompt_template are defined.
        Otherwise, default to concatenating required fields line by line.
        """
        sig = self.metadata.signature
        if sig and sig.prompt_template:
            return sig.prompt_template.format(**inputs)
        required = sig.required_inputs if sig else []
        parts = [str(inputs.get(r, "")) for r in required]
        return "\n".join(parts)

    def call_lm(self, prompt: str, lm: LMModule) -> str:
        """Convenience method for calling an LM module."""
        return lm(prompt=prompt)

    @abstractmethod
    def forward(self, inputs: T_in) -> T_out:
        """
        Eager path: transform `inputs` into outputs. Subclasses must implement.
        """
        pass

    def to_plan(self, inputs: T_in) -> Optional[ExecutionPlan]:
        """
        Optional: produce an ExecutionPlan for concurrency or distributed execution.
        Return None by default => no concurrency plan; just run .forward().
        """
        return None

    def combine_plan_results(self, results: Dict[str, Any], inputs: T_in) -> T_out:
        """
        If to_plan() returns multiple tasks, define how to merge the results
        into a final T_out. Default: if there’s exactly one task, return it;
        otherwise return the entire dict.
        """
        if len(results) == 1:
            return next(iter(results.values()))  # type: ignore
        return results  # type: ignore

    def get_signature(self) -> Optional[Signature]:
        """Utility for introspection or input validation."""
        return self.metadata.signature

    def __call__(self, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        sig = self.get_signature()
        if not sig:
            raise ValueError(f"Operator '{self.name}' is missing a signature.")

        validated_inputs: T_in = sig.validate_inputs(inputs)
        plan = self.to_plan(validated_inputs)
        if plan is not None:
            scheduler = Scheduler()
            results = scheduler.run_plan(plan)
            raw_output = self.combine_plan_results(results, validated_inputs)
        else:
            raw_output = self.forward(validated_inputs)

        from avior.core.trace_context import get_current_trace_context, TraceRecord
        ctx = get_current_trace_context()
        if ctx:
            record = TraceRecord(
                operator_name=self.name,
                operator_class=self.__class__.__name__,
                input_data=validated_inputs,
                output_data=raw_output,
            )
            ctx.add_record(record)

        validated_output: T_out = sig.validate_output(raw_output)
        return validated_output
