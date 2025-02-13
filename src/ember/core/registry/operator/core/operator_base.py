from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine.xcs_engine import XCSScheduler, compile_graph
from ember.core.registry.model.modules.lm import LMModule
from ember.core.registry.prompt_signature.signatures import Signature
from ember.xcs.tracer.xcs_tracing import TraceRecord, get_current_trace_context

# Type variables for input and output models.
T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class OperatorMetadata(BaseModel):
    """Metadata for an operator used in introspection and registry.

    Attributes:
        code (str): Unique identifier for the operator.
        description (str): A brief explanation of the operator's purpose.
        signature (Optional[Signature]): Optional signature instance for input/output validation.
    """

    code: str
    description: str
    signature: Optional[Signature] = None

    class Config:
        frozen = True  # Makes metadata immutable once created.


class Operator(ABC, Generic[T_in, T_out]):
    """Abstract base class for operators with automatic sub-operator registration.

    This class supports both synchronous execution via the 'forward' method and
    asynchronous/concurrent execution via a concurrency subgraph. Input and output
    validation are delegated to the associated Signature instance.

    Attributes:
        metadata (OperatorMetadata): Contains introspection details (code, description, and signature).
        name (str): A human-readable name for the operator.
        lm_modules (List[LMModule]): The list of language model modules for LLM calls.
    """

    # Default operator metadata. Subclasses should override as needed.
    metadata: OperatorMetadata = OperatorMetadata(
        code="BASE",
        description="Base operator",
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
        """Initialize an Operator instance.

        Creates an instance-specific copy of the operator metadata to avoid shared state.
        Optionally registers language model modules and sub-operators.

        Args:
            name (str): Human-readable identifier for the operator.
            lm_modules (Optional[List[LMModule]]): List of LMModule instances for language model calls.
            sub_operators (Optional[List[Operator[Any, Any]]]): List of sub-operators to be registered automatically.
            signature (Optional[Signature]): Signature for input/output validation that overrides the default.
        """
        object.__setattr__(
            self, "_sub_operators", {}
        )  # type: Dict[str, Operator[Any, Any]]
        self.metadata = self.__class__.metadata.model_copy(deep=True)
        self.name = name
        self.lm_modules = lm_modules if lm_modules is not None else []

        if signature is not None:
            self.metadata.signature = signature

        # TODO (jaredquincy): Revisit how sub-operators are named, registered, and stored.
        if sub_operators is not None:
            for index, sub_operator in enumerate(sub_operators):
                setattr(self, f"sub_op_{index}", sub_operator)

    def __setattr__(self, attr_name: str, value: Any) -> None:
        """Set an attribute and auto-register Operator instances.

        If the provided value is an Operator (and the attribute name is not '__call__'),
        it gets added to the sub-operator registry.

        Args:
            attr_name (str): The name of the attribute.
            value (Any): The value to assign.
        """
        if attr_name != "__call__" and isinstance(value, Operator):
            self._sub_operators[attr_name] = value
        super().__setattr__(attr_name, value)

    @property
    def sub_operators(self) -> Dict[str, Operator[Any, Any]]:
        """Obtain the registered sub-operators.

        Returns:
            Dict[str, Operator[Any, Any]]: Mapping of attribute names to sub-operator instances.
        """
        return getattr(self, "_sub_operators", {})

    def call_lm(self, *, prompt: str, lm: LMModule) -> str:
        """Invoke the language model with a provided prompt.

        Args:
            prompt (str): The text prompt to process.
            lm (LMModule): An LMModule instance that processes the prompt.

        Returns:
            str: The response from the language model.
        """
        return lm(prompt=prompt)

    @abstractmethod
    def forward(self, *, inputs: T_in) -> T_out:
        """Execute the primary computation of the operator.

        Subclasses must implement this method to transform validated input into the corresponding output.

        Args:
            inputs (T_in): A validated input model instance.

        Returns:
            T_out: The validated output model instance.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def to_plan(self, *, inputs: T_in) -> Optional[XCSGraph]:
        """Generate an optional execution plan for concurrent or distributed execution.

        If an execution plan is provided, the operator's computation may run concurrently;
        otherwise, the 'forward' method is executed synchronously.

        Args:
            inputs (T_in): A validated input model instance.

        Returns:
            Optional[XCSGraph]: The execution plan if applicable; otherwise, None.
        """
        return None

    def combine_plan_results(self, *, results: Dict[str, Any], inputs: T_in) -> T_out:
        """Combine results from concurrent tasks into a single output.

        If exactly one result is present, it is returned directly; otherwise, the full results
        dictionary is passed through after being cast to T_out.

        Args:
            results (Dict[str, Any]): Mapping of task identifiers to their outputs.
            inputs (T_in): The original validated input model instance.

        Returns:
            T_out: The consolidated output model instance.
        """
        if len(results) == 1:
            return cast(T_out, next(iter(results.values())))
        return cast(T_out, results)

    def get_signature(self) -> Optional[Signature]:
        """Get the signature associated with this operator.

        Returns:
            Optional[Signature]: The Signature instance if available; otherwise, None.
        """
        return self.metadata.signature

    @classmethod
    def build_inputs(cls, **fields: Any) -> T_in:
        """Construct an input model instance based on the operator's signature.

        Args:
            **fields: Arbitrary keyword arguments that correspond to the fields of the input model.

        Returns:
            T_in: The constructed input model instance.

        Raises:
            NotImplementedError: If no input model is defined for the operator.
        """
        if cls.metadata.signature and cls.metadata.signature.input_model:
            return cls.metadata.signature.input_model(**fields)
        raise NotImplementedError("No input model defined for operator.")

    def _validate_inputs(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_in:
        """Validate and convert input data using the operator's signature.

        Args:
            inputs (Union[T_in, Dict[str, Any]]): Input data as a model instance or as a dictionary.

        Returns:
            T_in: The validated input model instance.

        Raises:
            ValueError: If the operator's signature is missing.
        """
        signature: Optional[Signature] = self.get_signature()
        if signature is None:
            raise ValueError(f"Operator '{self.name}' is missing a signature.")
        return signature.validate_inputs(inputs=inputs)

    def _execute_plan_or_forward(self, *, validated_inputs: T_in) -> Any:
        """Execute the operator using an execution plan if available; otherwise, invoke 'forward'.

        Args:
            validated_inputs (T_in): The validated input model instance.

        Returns:
            Any: The raw output from the execution plan or the 'forward' computation.
        """
        execution_plan: Optional[XCSGraph] = self.to_plan(inputs=validated_inputs)
        if execution_plan is not None:
            scheduler: XCSScheduler = XCSScheduler()
            results: Dict[str, Any] = scheduler.run_plan(plan=execution_plan)
            return self.combine_plan_results(results=results, inputs=validated_inputs)
        return self.forward(inputs=validated_inputs)

    def _record_trace(self, *, validated_inputs: T_in, raw_output: Any) -> None:
        """Record a trace of the operator's execution if a trace context is available.

        Args:
            validated_inputs (T_in): The input model instance used during execution.
            raw_output (Any): The raw output produced by the execution.
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
            raw_output (Any): The raw output data from execution.

        Returns:
            T_out: The validated output model instance.

        Raises:
            ValueError: If the signature is missing during output validation.
        """
        signature: Optional[Signature] = self.get_signature()
        if signature is None:
            raise ValueError("Missing signature during output validation.")
        return signature.validate_output(raw_output=raw_output)

    def __call__(self, *, inputs: Union[T_in, Dict[str, Any]]) -> T_out:
        """Entry point for operator execution.

        Validates inputs, optionally builds a concurrency subgraph (if supported), and
        executes the operator. Execution trace is recorded before returning validated output.

        Args:
            inputs (Union[T_in, Dict[str, Any]]): Input data as a model instance or dictionary.

        Returns:
            T_out: The validated output model instance.
        """
        validated_inputs: T_in = self._validate_inputs(inputs=inputs)
        # Convert validated inputs to dict for reference if needed.
        as_dict: Dict[str, Any] = (
            validated_inputs.model_dump()
            if hasattr(validated_inputs, "model_dump")
            else dict(validated_inputs)
        )

        # Attempt to build a concurrency plan
        execution_plan: Optional[XCSGraph] = self.to_plan(inputs=validated_inputs)
        if execution_plan is not None:
            scheduler: XCSScheduler = XCSScheduler()
            results: Dict[str, Any] = scheduler.run_plan(
                plan=compile_graph(graph=execution_plan)
            )
            raw_output = self.combine_plan_results(
                results=results, inputs=validated_inputs
            )
        else:
            # Synchronous fallback
            raw_output = self.forward(inputs=validated_inputs)

        self._record_trace(validated_inputs=validated_inputs, raw_output=raw_output)
        return self._validate_output(raw_output=raw_output)
