import functools
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from src.ember.xcs.scheduler import Scheduler
from src.ember.xcs.tracer import convert_traced_graph_to_plan
from ember.src.ember.registry.operator.core.operator_base import Operator

T = TypeVar("T", bound=Operator)


def jit(sample_input: Optional[Dict[str, Any]] = None) -> Callable[[Type[T]], Type[T]]:
    """Decorator that JIT-traces an Operator subclass.

    If a sample input is provided, the execution plan is eagerly compiled during
    initialization; otherwise, a lazy trace is performed upon the first call.

    Args:
        sample_input (Optional[Dict[str, Any]]): Optional sample input data for eager plan compilation.

    Returns:
        Callable[[Type[T]], Type[T]]: A class decorator that modifies the __init__ and __call__
            methods of an Operator subclass to enable JIT tracing.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, Operator):
            raise TypeError("@jit can only be applied to an Operator subclass.")

        original_init = cls.__init__
        original_call = cls.__call__

        @functools.wraps(original_init)
        def new_init(self: T, *args: Any, **kwargs: Any) -> None:
            """Modified initializer that sets up JIT tracing attributes.

            Args:
                *args (Any): Positional arguments passed to the original __init__.
                **kwargs (Any): Keyword arguments passed to the original __init__.
            """
            original_init(self, *args, **kwargs)
            self._compiled_plan: Optional[Any] = None
            self._jit_traced: bool = False
            if sample_input is not None:
                # Eagerly compile the plan using the provided sample input.
                self._trace_and_compile(sample_input)

        def jit_call(self: T, inputs: Any) -> Any:
            """Modified __call__ method that validates inputs, compiles the plan if needed,
            and executes the JIT-compiled operator plan.

            Args:
                inputs (Any): The input data for the operator.

            Returns:
                Any: The output from the execution, validated by the operator's signature.
            """
            validated_inputs: Any = self.get_signature().validate_inputs(inputs)

            if not self._jit_traced:
                # Compile the execution plan using the validated inputs.
                self._trace_and_compile(validated_inputs)
                self._jit_traced = True

            if self._compiled_plan is not None:
                # Run the compiled execution plan.
                scheduler: Scheduler = Scheduler()
                results: Any = scheduler.run_plan(self._compiled_plan)
                merged_results: Any = self.combine_plan_results(
                    results, validated_inputs
                )
                validated_output: Any = self.get_signature().validate_output(
                    merged_results
                )
                return validated_output
            else:
                # Fallback to the original operator call.
                raw_output: Any = original_call(self, validated_inputs)
                return self.get_signature().validate_output(raw_output)

        def trace_and_compile(self: T, sample_in: Any) -> None:
            """Runs the tracer to build an execution plan and caches it.

            This method prevents re-entrant tracing and skips execution if tracing has
            already been performed.

            Args:
                sample_in (Any): The sample input data used for tracing.
            """
            if self._jit_traced:
                return

            if getattr(self, "_in_tracing", False):
                return

            self._in_tracing = True
            try:
                from src.ember.xcs.tracer.trace_context import (
                    get_current_trace_context,
                )

                if get_current_trace_context():
                    return

                from src.ember.xcs.tracer import (
                    TracerContext,
                    convert_traced_graph_to_plan,
                )

                with TracerContext(self, sample_in) as tgraph:
                    tracer_graph: Any = tgraph.run_trace()
                plan: Any = convert_traced_graph_to_plan(tracer_graph)
                self._compiled_plan = plan
                self._jit_traced = True
            finally:
                self._in_tracing = False

        setattr(cls, "__init__", new_init)
        setattr(cls, "__call__", jit_call)
        setattr(cls, "_trace_and_compile", trace_and_compile)

        return cls

    return decorator
