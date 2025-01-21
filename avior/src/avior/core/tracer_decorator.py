# tracer_decorator.py

import functools
from typing import Any, Dict, Optional

from src.avior.core.scheduler import Scheduler
from src.avior.core.tracer import convert_traced_graph_to_plan
from src.avior.registry.operator.operator_base import Operator

def jit(sample_input: Optional[Dict[str, Any]] = None):
    """
    A decorator that JIT-traces an Operator:
      - If sample_input is provided, we immediately build the plan in __init__
      - Otherwise, we do a lazy trace on the first call
    """
    def decorator(cls):
        if not issubclass(cls, Operator):
            raise TypeError("@jit can only be applied to an Operator subclass.")

        # We intercept the __init__ and __call__ of the class
        original_init = cls.__init__
        original_call = cls.__call__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._compiled_plan = None
            self._jit_traced = False
            if sample_input is not None:
                # Eagerly build the plan right away
                self._trace_and_compile(sample_input)

        def jit_call(self, inputs: Any):
            # Always validate inputs just like Operator.__call__
            validated_inputs = self.get_signature().validate_inputs(inputs)

            if not self._jit_traced:
                # Use validated_inputs for the trace
                self._trace_and_compile(validated_inputs)
                self._jit_traced = True

            if self._compiled_plan:
                # Run the JIT-compiled plan
                scheduler = Scheduler()
                results = scheduler.run_plan(self._compiled_plan)
                merged = self.combine_plan_results(results, validated_inputs)
                validated_output = self.get_signature().validate_output(merged)
                return validated_output
            else:
                # Fallback => call the normal operator path, so we actually run forward()
                # exactly as the base class does.
                raw_output = original_call(self, validated_inputs)
                return self.get_signature().validate_output(raw_output)

        def trace_and_compile(self, sample_in: Any):
            """
            The function that runs the tracer, then builds an ExecutionPlan or IR
            and stores it in self._compiled_plan.
            """
            if self._jit_traced:
                return
            
            # Guard to prevent re-entrancy
            if getattr(self, "_in_tracing", False):
                return
            
            setattr(self, "_in_tracing", True)
            try:
                from src.avior.core.trace_context import get_current_trace_context
                if get_current_trace_context():
                    return

                from avior.core.tracer import TracerContext, convert_traced_graph_to_plan
                with TracerContext(self, sample_in) as tgraph:
                    tracer_graph = tgraph.run_trace()
                plan = convert_traced_graph_to_plan(tracer_graph)
                self._compiled_plan = plan
                self._jit_traced = True
            finally:
                setattr(self, "_in_tracing", False)

        # Attach the new init and call
        setattr(cls, "__init__", new_init)
        setattr(cls, "__call__", jit_call)
        setattr(cls, "_trace_and_compile", trace_and_compile)

        return cls
    return decorator