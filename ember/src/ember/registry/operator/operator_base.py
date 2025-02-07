from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import Any, Callable, cast, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel
from src.ember.core.scheduler import ExecutionPlan, Scheduler
from src.ember.modules.lm_modules import LMModule
from src.ember.registry.prompt_signature.signatures import Signature
from ember.src.ember.core.tracer.trace_context import (
    TraceRecord,
    get_current_trace_context,
)

# Type variables for input and output models.
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class OperatorType(Enum):
    """Enumeration for operator topologies or roles.

    Attributes:
        RECURRENT: Indicates a recurrent operator, shape preserving.
        FAN_OUT: Indicates a fan-out operator, shape expanding.
        FAN_IN: Indicates a fan-in operator, shape contracting.
    """

    RECURRENT = auto()
    FAN_OUT = auto()
    FAN_IN = auto()


class OperatorMetadata(BaseModel):
    """Metadata associated with an operator for introspection and registry.

    Attributes:
        code (str): A unique identifier for the operator.
        description (str): A brief explanation of the operator's purpose.
        operator_type (OperatorType): The topology or role of the operator.
        signature (Optional[Signature]): An optional Signature instance for input/output validation.
    """

    code: str
    description: str
    operator_type: OperatorType
    signature: Optional[Signature] = None


def run_in_parallel(
    function: Callable[..., Any],
    args_list: List[Dict[str, Any]],
    max_workers: Optional[int] = None,
) -> List[Any]:
    """Run a function in parallel using a thread pool.

    This utility submits concurrent invocations of the provided function and aggregates
    the resulting outputs.

    Args:
        function: A callable that accepts keyword arguments.
        args_list: A list of dictionaries, each representing the keyword arguments for a
            single function call.
        max_workers: The maximum number of threads to use. If None, a default value is used.

    Returns:
        List[Any]: A list containing the outputs from each function call.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(function, **args) for args in args_list]
        results: List[Any] = [future.result() for future in futures]
    return results


class Operator(ABC, Generic[T_in, T_out]):
    """Abstract base class for operators with auto-registration of sub-operators.

    This class supports both direct forward execution and the construction of concurrency
    plans. Input and output validation is delegated to an associated Signature instance.

    Attributes:
        metadata (OperatorMetadata): Contains introspection details such as code, description,
            operator type, and the input/output signature.
        name (str): A human-friendly name for the operator.
        lm_modules (List[LMModule]): A list of LMModule instances for performing language model calls.
    """

    # Default operator metadata. Subclasses should override as needed.
    metadata: OperatorMetadata = OperatorMetadata(
        code="BASE",
        description="Base operator with sub-operator auto-registration.",
        operator_type=OperatorType.RECURRENT,
        signature=Signature(required_inputs=[]),
    )

    def __init__(
        self,
        *,
        name: str = "Operator",
        lm_modules: Optional[List[LMModule]] = None,
        sub_operators: Optional[List[Operator[Any, Any]]] = None,
        signature: Optional[Signature] = None,
    ) -> None:
        """Initialize the operator.

        An instance-specific copy of the class metadata is created to prevent shared mutable state.
        Optionally, provided sub-operators are auto-registered.

        Args:
            name: A human-readable identifier for the operator.
            lm_modules: A list of LMModule instances for executing language model calls.
            sub_operators: Optional list of child operators to register automatically.
            signature: Optional Signature to validate input and output data.
        """
        object.__setattr__(
            self, "_sub_operators", {}
        )  # type: Dict[str, Operator[Any, Any]]
        self.metadata = self.__class__.metadata.copy(deep=True)
        self.name = name
        self.lm_modules = lm_modules if lm_modules is not None else []

        if signature is not None:
            self.metadata.signature = signature

        if sub_operators:
            for index, sub_operator in enumerate(sub_operators):
                setattr(self, f"sub_op_{index}", sub_operator)

    def __setattr__(self, attr_name: str, value: Any) -> None:
        """Assign an attribute with auto-registration for Operator instances.

        If the assigned value is an Operator instance (and the attribute name is not '__call__'),
        it is automatically recorded in the sub-operator registry.

        Args:
            attr_name: The name of the attribute.
            value: The value to assign.
        """
        if attr_name != "__call__" and isinstance(value, Operator):
            self._sub_operators[attr_name] = value
        super().__setattr__(attr_name, value)

    @property
    def sub_operators(self) -> Dict[str, Operator[Any, Any]]:
        """Return the registered sub-operators.

        Returns:
            Dict[str, Operator[Any, Any]]: A dictionary mapping attribute names to sub-operator instances.
        """
        return getattr(self, "_sub_operators", {})

    def build_prompt(self, *, inputs: Dict[str, Any]) -> str:
        """Build a prompt from the provided inputs using the operator's signature.

        If a prompt template is defined in the signature, it is formatted with the inputs.
        Otherwise, the required fields are concatenated line by line.

        Args:
            inputs: A dictionary containing input values.

        Returns:
            str: A formatted prompt string.
        """
        signature_obj: Optional[Signature] = self.metadata.signature
        if signature_obj is not None and signature_obj.prompt_template:
            return signature_obj.prompt_template.format(**inputs)

        required_fields: List[str] = (
            signature_obj.required_inputs if signature_obj else []
        )
        prompt_parts: List[str] = [
            str(inputs.get(field, "")) for field in required_fields
        ]
        return "\n".join(prompt_parts)

    def call_lm(self, *, prompt: str, lm: LMModule) -> str:
        """Call the language model with the provided prompt.

        Args:
            prompt: The text prompt to be processed.
            lm: An LMModule instance that will process the prompt.

        Returns:
            str: The response from the language model.
        """
        return lm(prompt=prompt)

    @abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Execute the primary computation of the operator.

        Subclasses must implement this method to transform validated inputs into outputs.

        Args:
            inputs: A validated input model instance.

        Returns:
            T_out: A validated output model instance.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def to_plan(self, *, inputs: T_in) -> Optional[ExecutionPlan]:
        """Generate an optional execution plan for concurrent or distributed execution.

        If an execution plan is provided, the operator's forward computation may be executed concurrently;
        otherwise, the forward method is invoked directly.

        Args:
            inputs: A validated input model instance.

        Returns:
            Optional[ExecutionPlan]: An execution plan if a concurrency strategy is applicable; otherwise, None.
        """
        return None

    def combine_plan_results(self, *, results: Dict[str, Any], inputs: T_in) -> T_out:
        """Combine results from multiple concurrent tasks into a single output.

        If exactly one result is available, it is returned directly. Otherwise, the entire results dictionary
        is passed along for further processing.

        Args:
            results: A dictionary mapping task identifiers to their outputs.
            inputs: The original validated input model instance.

        Returns:
            T_out: A consolidated output model instance.
        """
        if len(results) == 1:
            return cast(T_out, next(iter(results.values())))
        return cast(T_out, results)

    def get_signature(self) -> Optional[Signature]:
        """Retrieve the signature associated with this operator.

        Returns:
            Optional[Signature]: The Signature instance if available; otherwise, None.
        """
        return self.metadata.signature

    def __call__(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Execute the operator using the provided inputs.

        This method performs input validation, optionally executes a concurrency plan,
        and records tracing information if available.

        Args:
            inputs: Either a validated input model instance or a dictionary of raw input data.

        Returns:
            T_out: A validated output model instance after processing.

        Raises:
            ValueError: If the operator does not have an associated signature.
        """
        signature_obj: Optional[Signature] = self.get_signature()
        if signature_obj is None:
            raise ValueError(f"Operator '{self.name}' is missing a signature.")

        validated_inputs: T_in = signature_obj.validate_inputs(inputs=inputs)
        execution_plan: Optional[ExecutionPlan] = self.to_plan(inputs=validated_inputs)
        if execution_plan is not None:
            scheduler = Scheduler()
            results: Dict[str, Any] = scheduler.run_plan(plan=execution_plan)
            raw_output: Any = self.combine_plan_results(
                results=results, inputs=validated_inputs
            )
        else:
            raw_output = self.forward(inputs=validated_inputs)

        current_trace = get_current_trace_context()
        if current_trace is not None:
            trace_record = TraceRecord(
                operator_name=self.name,
                operator_class=self.__class__.__name__,
                input_data=validated_inputs,
                output_data=raw_output,
            )
            current_trace.add_record(record=trace_record)

        validated_output: T_out = signature_obj.validate_output(raw_output=raw_output)
        return validated_output
