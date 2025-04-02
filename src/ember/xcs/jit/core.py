"""Core JIT compilation system for XCS.

Provides the unified JIT decoration mechanism that serves as the entry point
for all JIT operations. This module implements strategy selection logic and
manages the overall JIT compilation process.

The implementation follows classic functional principles with proper
separation of concerns between selection, compilation, and caching.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from ember.xcs.jit.cache import JITCache, get_cache
from ember.xcs.jit.modes import JITMode
from ember.xcs.jit.strategies import Strategy

# Type variable for function/class being decorated
F = TypeVar('F', bound=Callable)

# Setup logging for the JIT system
logger = logging.getLogger(__name__)


class JITSettings:
    """Settings for JIT compilation behavior.
    
    Encapsulates all configuration options for the JIT system.
    """
    
    def __init__(
        self,
        mode: Union[JITMode, str] = JITMode.AUTO,
        force_trace: bool = False,
        sample_input: Optional[Dict[str, Any]] = None,
        custom_cache: Optional[JITCache] = None,
        recursive: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize JIT settings.
        
        Args:
            mode: JIT compilation mode to use
            force_trace: Whether to force retracing
            sample_input: Optional sample input for eager compilation
            custom_cache: Custom cache instance
            recursive: Whether to apply JIT recursively
            **kwargs: Additional strategy-specific options
        """
        # Normalize mode to enum
        if isinstance(mode, str):
            try:
                self.mode = JITMode(mode.lower())
            except ValueError:
                logger.warning(f"Unknown JIT mode '{mode}', falling back to AUTO")
                self.mode = JITMode.AUTO
        else:
            self.mode = mode
        
        self.force_trace = force_trace
        self.sample_input = sample_input
        self.custom_cache = custom_cache
        self.recursive = recursive
        self.options = kwargs


class StrategySelector:
    """Selector for choosing the appropriate JIT strategy.
    
    Implements policy for selecting the most appropriate strategy based on
    function characteristics and user preferences.
    """
    
    def __init__(self) -> None:
        """Initialize the strategy selector."""
        # Import strategies here to avoid circular dependencies
        from ember.xcs.jit.strategies.enhanced import EnhancedStrategy
        from ember.xcs.jit.strategies.structural import StructuralStrategy
        from ember.xcs.jit.strategies.trace import TraceStrategy
        
        # Map of mode to strategy classes
        self._strategies: Dict[JITMode, Strategy] = {
            JITMode.TRACE: TraceStrategy(),
            JITMode.STRUCTURAL: StructuralStrategy(),
            JITMode.ENHANCED: EnhancedStrategy(),
        }
    
    def select_strategy(
        self, 
        func: Callable[..., Any], 
        mode: JITMode = JITMode.AUTO
    ) -> Strategy:
        """Select the most appropriate strategy for a function.
        
        Args:
            func: Function to analyze
            mode: User-specified mode, or AUTO for auto-selection
            
        Returns:
            Selected strategy instance
        """
        # Return explicit strategy if specified
        if mode != JITMode.AUTO:
            return self._strategies[mode]
        
        # Auto-selection based on function analysis
        results = []
        for strategy_mode, strategy in self._strategies.items():
            analysis = strategy.analyze(func)
            results.append((strategy_mode, strategy, analysis))
        
        # Sort by score in descending order
        results.sort(key=lambda x: x[2].get("score", 0), reverse=True)
        
        # Log detailed analysis for debugging
        for mode, _, analysis in results:
            logger.debug(
                f"Strategy {mode.value}: score={analysis.get('score', 0)}, "
                f"rationale={analysis.get('rationale', 'No rationale provided')}"
            )
        
        # Return the highest-scoring strategy
        return results[0][1]


# Global strategy selector
_selector = StrategySelector()


def _jit_function(
    func: Callable[..., Any],
    strategy: Strategy,
    settings: JITSettings
) -> Callable[..., Any]:
    """Compiles a regular function using the chosen JIT strategy.
    
    Args:
        func: Function to compile
        strategy: Selected compilation strategy
        settings: JIT configuration settings
        
    Returns:
        Compiled function
    """
    return strategy.compile(
        func,
        sample_input=settings.sample_input,
        force_trace=settings.force_trace,
        recursive=settings.recursive,
        cache=settings.custom_cache or get_cache(),
        **settings.options
    )


def _create_operator_forward_proxy(strategy: Strategy, settings: JITSettings):
    """Creates a specialized proxy for operator's forward method.
    
    Instead of compiling the forward method directly, this creates a proxy function
    that correctly maintains the instance context when called.
    
    Args:
        strategy: JIT strategy to use
        settings: JIT configuration settings
        
    Returns:
        Function that creates a callable forward proxy for an operator instance
    """
    def create_forward_proxy(instance, forward_method):
        """Creates a bound method proxy that preserves instance context.
        
        Args:
            instance: Operator instance
            forward_method: The forward method to proxy
            
        Returns:
            Callable that acts like the operator's forward method
        """
        # Create a closure to capture the instance and method
        def forward_proxy(*, inputs):
            """Execute the forward method with tracing support.
            
            This proxy preserves the call interface of the original forward method
            while adding tracing transparency and JIT optimizations.
            """
            # Import here to avoid circular dependencies
            from ember.xcs.tracer.xcs_tracing import TracerContext
            
            # Check if we're in a tracing context
            tracer = TracerContext.get_current()
            
            # If we're in a tracing context, track the call
            if tracer and tracer.is_active:
                # Use the instance's name if available, otherwise use class name
                operator_name = getattr(instance, "name", instance.__class__.__name__)
                # Track the call with proper operator identity
                call_id = tracer.track_call(instance, inputs)
                
                try:
                    # Directly invoke the instance's forward method
                    result = forward_method(instance, inputs=inputs)
                    # Record the successful call completion
                    tracer.complete_call(call_id, result)
                    return result
                except Exception as e:
                    # Record the exception for observability
                    tracer.complete_call(call_id, {}, e)
                    # Re-raise the original exception to maintain behavior
                    raise
            else:
                # No tracing context - direct execution path
                return forward_method(instance, inputs=inputs)
            
        return forward_proxy
    
    return create_forward_proxy
    

def _jit_operator_class(
    cls: Type,
    strategy: Strategy,
    settings: JITSettings
) -> Type:
    """Creates a JIT-optimized version of an operator class.
    
    This function isolates the operator-specific JIT logic, creating a new
    class that inherits from the original but with optimized execution.
    
    Args:
        cls: Operator class to optimize
        strategy: Selected compilation strategy
        settings: JIT configuration settings
        
    Returns:
        JIT-optimized operator class
    """
    class_name = cls.__name__ + "_JIT"
    strategy_name = strategy.__class__.__name__.replace("Strategy", "")
    
    # Verify the class has a forward method
    if not hasattr(cls, "forward"):
        raise ValueError(f"Operator class {cls.__name__} must have a forward method")
    
    # Get the forward method - we'll use it directly in call
    original_forward = cls.forward
    
    # Create a forward proxy factory - used to handle binding self properly
    create_proxy = _create_operator_forward_proxy(strategy, settings)
    
    # We compile the full operation inside the __call__ method, not just forward
    def jit_init(self, *args, **kwargs):
        # Filter out 'inputs' parameter which belongs to __call__, not __init__
        init_kwargs = {k: v for k, v in kwargs.items() if k != 'inputs'}
        # Initialize the class normally
        cls.__init__(self, *args, **init_kwargs)
        # Create a proxy and compile it - this happens per instance
        self._forward_proxy = create_proxy(self, original_forward)
        self._compiled_func = strategy.compile(
            self._forward_proxy,
            sample_input=settings.sample_input,
            force_trace=settings.force_trace,
            recursive=settings.recursive,
            cache=settings.custom_cache or get_cache(),
            **settings.options
        )
    
    def jit_call(self, **kwargs):
        # Get required inputs - everything else is passed to the compiled function as-is
        inputs = kwargs.get("inputs", {})
        
        # Import here to avoid circular dependencies
        from ember.xcs.tracer.xcs_tracing import TracerContext
        
        # Get current tracing context to properly propagate tracing through call chain
        tracer = TracerContext.get_current()
        
        # If we're in a trace context, add operator information before execution
        if tracer and tracer.is_active:
            # Track the operator call in the trace context
            call_id = tracer.track_call(self, inputs)
            
            try:
                # Execute using compiled function
                result = self._compiled_func(inputs=inputs)
                
                # Complete the trace record with successful execution
                tracer.complete_call(call_id, result)
                return result
            except Exception as e:
                # Record the exception but don't swallow it
                tracer.complete_call(call_id, {}, e)
                raise
        else:
            # Not tracing - direct execution path
            return self._compiled_func(inputs=inputs)
    
    # Create the JIT-optimized class
    return type(class_name, (cls,), {
        "__init__": jit_init,
        "__call__": jit_call,
        "_jit_strategy": strategy_name,
        "__doc__": cls.__doc__
    })


def jit(
    func: Optional[Callable[..., Any]] = None,
    *,
    mode: Union[str, JITMode] = JITMode.AUTO,
    force_trace: bool = False,
    sample_input: Optional[Dict[str, Any]] = None,
    cache: Optional[JITCache] = None,
    recursive: bool = True,
    **kwargs: Any
) -> Any:
    """Just-In-Time compilation decorator for XCS.
    
    Optimizes functions and operators using Just-In-Time compilation with
    various strategies. It automatically selects the most appropriate
    strategy based on the function's characteristics, or uses an explicitly
    specified one.
    
    Args:
        func: Function or class to optimize
        mode: JIT mode to use (auto, trace, structural, enhanced)
        force_trace: Whether to force retracing even if cached
        sample_input: Optional sample input for eager compilation
        cache: Optional custom cache to use
        recursive: Whether to apply JIT recursively to nested functions
        **kwargs: Additional options passed to the selected strategy
        
    Returns:
        Optimized function or class
        
    Example:
        ```python
        @jit
        class MyOperator(Operator):
            def forward(self, *, inputs):
                return process(inputs)
                
        # With explicit strategy
        @jit(mode="structural", sample_input={"data": "example"})
        def process_data(*, inputs):
            return {"result": complex_calculation(inputs["data"])}
        ```
    """
    # Handle both decorator styles (@jit and @jit())
    if func is None:
        return lambda f: jit(
            f,
            mode=mode,
            force_trace=force_trace,
            sample_input=sample_input,
            cache=cache,
            recursive=recursive,
            **kwargs
        )
    
    # Configure settings
    settings = JITSettings(
        mode=mode,
        force_trace=force_trace,
        sample_input=sample_input,
        custom_cache=cache,
        recursive=recursive,
        **kwargs
    )
    
    # Select appropriate strategy
    strategy = _selector.select_strategy(func, settings.mode)
    
    # Dispatch based on target type - function or operator class
    is_operator = inspect.isclass(func) and hasattr(func, "forward")
    if is_operator:
        return _jit_operator_class(func, strategy, settings)
    else:
        return _jit_function(func, strategy, settings)


def get_jit_stats(func: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Get statistics about JIT compilation and execution.
    
    Args:
        func: Optional function to get stats for. If None, returns overall stats.
        
    Returns:
        Dictionary with compilation and execution statistics
    """
    cache = get_cache()
    return cache.get_metrics(func)


def explain_jit_selection(func: Callable[..., Any]) -> Dict[str, Any]:
    """Explain why a particular JIT strategy would be selected.
    
    Useful for understanding and debugging the auto-selection process.
    
    Args:
        func: Function to analyze
        
    Returns:
        Dictionary with detailed analysis from each strategy
    """
    from ember.xcs.jit.strategies.enhanced import EnhancedStrategy
    from ember.xcs.jit.strategies.structural import StructuralStrategy
    from ember.xcs.jit.strategies.trace import TraceStrategy
    
    strategies = {
        "trace": TraceStrategy(),
        "structural": StructuralStrategy(),
        "enhanced": EnhancedStrategy(),
    }
    
    results = {}
    for name, strategy in strategies.items():
        results[name] = strategy.analyze(func)
    
    return results