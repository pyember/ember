from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel
from ember.xcs.scheduler import ExecutionPlan, Scheduler
from ember.core.registry.model.core.modules.lm_modules import LMModule
from ember.core.registry.prompt_signature.signatures import Signature
from ember.xcs.tracer.trace_context import TraceRecord, get_current_trace_context

# Type variables for input and output models.
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class OperatorMetadata(BaseModel):
    """Metadata associated with an operator for introspection and registry.

    Attributes:
        code (str): Unique identifier for the operator.
        description (str): Brief explanation of the operator's purpose.
        signature (Optional[Signature]): Optional Signature instance for input/output validation.
    """

    code: str
    description: str
    signature: Optional[Signature] = None

    class Config:
        frozen = True  # Ensures the metadata is immutable.


class Operator(ABC, Generic[T_in, T_out]):
    """Abstract base class for operators with auto-registration of sub-operators.

    This base class supports both direct execution (via the forward method) and concurrent
    execution (via an execution plan). Input and output validation are delegated to an
    associated Signature instance.

    Attributes:
        metadata (OperatorMetadata): Contains introspection details such as code, description, and signature.
        name (str): Human-readable name for the operator.
        lm_modules (List[LMModule]): List of language model modules for performing language model calls.
    """

    # Default operator metadata. Subclasses should override as needed.
    metadata: OperatorMetadata = OperatorMetadata(
        code="BASE",
        description="Base operator with sub-operator auto-registration.",
        signature=Signature(),
    )

    def __init__(
        self,
        *,
        name: str = "Operator",
        lm_modules: Optional[List[LMModule]] = None,
        sub_operators: Optional[List[Operator[Any, Any]]] = None,
        signature: Optional[Signature] = None,
    ) -> None:
        """Initialize an Operator instance with optional language model modules and sub-operators.

        An instance‐specific copy of the class metadata is created to avoid shared mutable state.
        Provided sub‐operators are automatically registered.

        Args:
            name (str): Human-readable identifier for the operator.
            lm_modules (Optional[List[LMModule]]): List of LMModule instances for language model calls.
            sub_operators (Optional[List[Operator[Any, Any]]]): List of sub-operators to auto‐register.
            signature (Optional[Signature]): Signature to validate input/output, overriding the default.

        """
        object.__setattr__(
            self, "_sub_operators", {}
        )  # type: Dict[str, Operator[Any, Any]]
        self.metadata = self.__class__.metadata.model_copy(deep=True)
        self.name = name
        self.lm_modules = lm_modules if lm_modules is not None else []

        if signature is not None:
            self.metadata.signature = signature

        if sub_operators is not None:
            for index, sub_operator in enumerate(sub_operators):
                setattr(self, f"sub_op_{index}", sub_operator)

    def __setattr__(self, attr_name: str, value: Any) -> None:
        """Set an attribute while auto-registering any Operator instances.

        If the assigned value is an Operator (and the attribute name is not '__call__'),
        it is automatically added to the sub-operator registry.

        Args:
            attr_name (str): Name of the attribute.
            value (Any): Value to assign.
        """
        if attr_name != "__call__" and isinstance(value, Operator):
            self._sub_operators[attr_name] = value
        super().__setattr__(attr_name, value)

    @property
    def sub_operators(self) -> Dict[str, Operator[Any, Any]]:
        """Return the dictionary of registered sub-operators.

        Returns:
            Dict[str, Operator[Any, Any]]: Mapping of attribute names to sub-operator instances.
        """
        return getattr(self, "_sub_operators", {})

    def call_lm(self, *, prompt: str, lm: LMModule) -> str:
        """Invoke the language model with the provided prompt.

        Args:
            prompt (str): The text prompt to process.
            lm (LMModule): LMModule instance that processes the prompt.

        Returns:
            str: Language model response.
        """
        return lm(prompt=prompt)

    @abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Execute the primary computation of the operator.

        Subclasses must implement this method to transform validated input into output.

        Args:
            inputs (T_in): Validated input model instance.

        Returns:
            T_out: Validated output model instance.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def to_plan(self, *, inputs: T_in) -> Optional[ExecutionPlan]:
        """Generate an optional execution plan for concurrent or distributed execution.

        If an execution plan is provided, the operator’s computation may run concurrently;
        otherwise, the forward method is invoked directly.

        Args:
            inputs (T_in): Validated input model instance.

        Returns:
            Optional[ExecutionPlan]: Execution plan if applicable; otherwise, None.
        """
        return None

    def combine_plan_results(self, *, results: Dict[str, Any], inputs: T_in) -> T_out:
        """Combine results from concurrent tasks into a single output.

        If exactly one result is available, it is returned directly; otherwise, the entire results
        dictionary is passed along for further processing.

        Args:
            results (Dict[str, Any]): Mapping of task identifiers to their outputs.
            inputs (T_in): Original validated input model instance.

        Returns:
            T_out: Consolidated output model instance.
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

    @classmethod
    def build_inputs(cls, **fields: Any) -> T_in:
        """Construct an input model instance using the operator's signature.

        Args:
            **fields: Arbitrary keyword arguments matching the input model fields.

        Returns:
            T_in: Constructed input model instance.

        Raises:
            NotImplementedError: If no input model is defined for the operator.
        """
        if cls.metadata.signature and cls.metadata.signature.input_model:
            return cls.metadata.signature.input_model(**fields)
        raise NotImplementedError("No input model defined for operator.")

    def _validate_inputs(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_in:
        """Validate and convert input data using the operator's signature.

        Args:
            inputs (Union[T_in, Dict[str, Any]]): Input data as a model instance or dictionary.

        Returns:
            T_in: Validated input model instance.

        Raises:
            ValueError: If the operator's signature is missing.
        """
        signature_obj: Optional[Signature] = self.get_signature()
        if signature_obj is None:
            raise ValueError(f"Operator '{self.name}' is missing a signature.")
        return signature_obj.validate_inputs(inputs=inputs)

    def _execute_plan_or_forward(self, *, validated_inputs: T_in) -> Any:
        """Execute the operator using an execution plan if available; otherwise, invoke forward directly.

        Args:
            validated_inputs (T_in): Validated input model instance.

        Returns:
            Any: Raw output from the execution plan or forward computation.
        """
        execution_plan: Optional[ExecutionPlan] = self.to_plan(inputs=validated_inputs)
        if execution_plan is not None:
            scheduler: Scheduler = Scheduler()
            results: Dict[str, Any] = scheduler.run_plan(plan=execution_plan)
            return self.combine_plan_results(results=results, inputs=validated_inputs)
        return self.forward(inputs=validated_inputs)

    def _record_trace(self, *, validated_inputs: T_in, raw_output: Any) -> None:
        """Record a trace of the operator's execution if a trace context is available.

        Args:
            validated_inputs (T_in): Input model instance used in execution.
            raw_output (Any): Raw output resulting from execution.
        """
        current_trace = get_current_trace_context()
        if current_trace is not None:
            trace_record: TraceRecord = TraceRecord(
                operator_name=self.name,
                operator_class=self.__class__.__name__,
                input_data=validated_inputs,
                output_data=raw_output,
            )
            current_trace.add_record(record=trace_record)

    def _validate_output(self, *, raw_output: Any) -> T_out:
        """Validate and convert raw output using the operator's signature.

        Args:
            raw_output (Any): Raw output data from execution.

        Returns:
            T_out: Validated output model instance.

        Raises:
            ValueError: If the signature is missing during output validation.
        """
        signature_obj: Optional[Signature] = self.get_signature()
        if signature_obj is None:
            raise ValueError("Missing signature during output validation.")
        return signature_obj.validate_output(raw_output=raw_output)

    def __call__(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Invoke the operator with the provided input data.

        This method validates the inputs, executes the operator (using an execution plan
        if available), records the execution trace, and validates the output.

        Args:
            inputs (Union[T_in, Dict[str, Any]]): Input data as a model instance or dictionary.

        Returns:
            T_out: Validated output model instance.
        """
        validated_inputs: T_in = self._validate_inputs(inputs=inputs)
        raw_output: Any = self._execute_plan_or_forward(
            validated_inputs=validated_inputs
        )
        self._record_trace(validated_inputs=validated_inputs, raw_output=raw_output)
        return self._validate_output(raw_output=raw_output)
